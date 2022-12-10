from kerykeion import KrInstance, Report
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, element
from timezonefinder import TimezoneFinder
from sklearn.model_selection import train_test_split
import gen_random_datetime
import openai
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch import load, save, from_numpy, concat, Tensor

openai.api_key = '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'

# Citations: gen_random_datetime is courtesy of the developer Regis Santos, rg3915 on GitHub

# constants for astrology data processing
NUM_HOUSES = 12
SIGN_ARC = 30
DEFAULT_CHART_FEATURES = ['sun', 'moon', 'mercury', 'venus', 'mars', 
                          'jupiter', 'saturn', 'uranus', 'neptune', 'pluto', 
                          'asc', 'midheaven']
DEFAULT_NUM_CHART_FEATURES = len(DEFAULT_CHART_FEATURES)


class LitChartBioData(LightningDataModule):
    """
    Handle the preparation of datasets consisting of both numeric chart data and
    natural language bio data. Preprocess the bio data with the 
    BERTTokenizer by default. Basically a wrapper for the ChartBioData class
    for use wtih pyTorch Lightning.
    """
    def __init__(self, batch_size=100, num_bio_features = 1024, 
                 use_chart_features = DEFAULT_CHART_FEATURES, num_chart_features = DEFAULT_NUM_CHART_FEATURES, 
                 train_filepath = None, test_filepath = None):
        """
        Args:
            (int) batch_size : self-explanatory, desired batch_size for training
            (int) num_bio_features : for processing natural language data, number of features it takes on in tensor
            (List(str)) use_chart_features: chart features to use when reading in the data
            (int) num_chart_features : number of chart features!
            (str) train_filepath, test_filepath: if training and/or test data already encoded as tensors, filepaths to them!
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_bio_features = num_bio_features
        self.num_chart_features = num_chart_features
        self.num_features = self.num_bio_features + self.num_chart_features
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.use_chart_features = use_chart_features

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.train = ChartBioData('train', self.num_bio_features, self.use_chart_features, self.num_chart_features, self.train_filepath)
        elif stage == 'test':
            self.test = ChartBioData('test', self.num_bio_features, self.use_chart_features, self.num_chart_features, self.test_filepath)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8)


class ChartBioData(Dataset):
    """
    Read in and process the numeric and bio chart data from files.
    Works as a pandas-to-torch interface and also processes the text data 
    using the BERT pre-trained tokenizer.
    """
    def __init__(self, stage : str, max_token_len : int = 1024, use_chart_features = DEFAULT_CHART_FEATURES, num_chart_features = DEFAULT_NUM_CHART_FEATURES, tensor_filepath = None):
        """
        Args:
            (str) stage: training or testing, for reading in correct data
            (int) max_token_len: for normalizing tensor size, max tokens bio can be processed to; if filepaths are passed in, ignored
            (List(str)) use_chart_features: chart_features to use when reading data from file
            (int) num_chart_features: number of features to be read in from the chart data. EXPECTED TO CORRESPOND WITH len(chart_features)! 
                                      OR, if a filepath is passed in, this number is expected to refer to the number of chart features in that file's tensor!
            (str) tensor_filepath: if encoded data has already been stored to a tensor, just read it in
        """
        super().__init__()

        labels = pd.read_csv(f'{stage}_labels.csv')
        self.labels = from_numpy(labels.values)
        self.size = self.labels.shape[0]


        # if features are already in a file just read!

        if tensor_filepath:
            self.combined_features = load(tensor_filepath)
            self.num_chart_features = num_chart_features
            self.num_bio_features = self.combined_features.size(1) - self.num_chart_features
            self.chart_features = self.combined_features[:, :self.num_chart_features]
            self.bio_features = self.combined_features[:, self.num_chart_features:]
            return


        # if features are not already processed, read everything in and process!
        self.chart_features = pd.read_csv(f'{stage}_data.csv', usecols=use_chart_features)
        self.bio_features = pd.read_csv(f'{stage}_data.csv', usecols=['bios'])
        self.num_bio_features = max_token_len    # for encoding purposes
        self.num_chart_features = num_chart_features

        ############ Make everything a tensor! #####################
        self.chart_features = from_numpy(self.chart_features.values)    
        # tokenize and encode the language data before making it a tensor
        self.bio_features = self.convert_bios(self.bio_features.squeeze())
        
        # recombine chart and bio features
        self.combined_features = concat((self.chart_features, self.bio_features), 1)
        save(self.combined_features, f'{stage}_features_len_{self.max_tokens}_tensor.pt')


    def convert_bios(self, bios: pd.Series) -> Tensor:
        """
        Converts bios to tensors!
        Args: 
            (Series) bios: Pandas series of all the biographies. 
        Returns:
            (Tensor) encoded: tensor of all the encoded bios.
        """
        # tokenize!
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = self.tokenizer(bios.tolist(), truncation=True, padding='max_length', max_length=self.max_tokens, return_tensors='pt')
        return encoded['input_ids']

    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.combined_features[idx], self.labels[idx]



###### Below are all the different methods that were used to read in and randomly generate parts of the dataset. #######


def read_charts(birth_data: pd.DataFrame):
    """
    Helper: use Kerykeion to read a single chart from the databank, then store 
    it in the dataframe.

    Args:
        (DataFrame) birth_data: df with birth data, sorted in each row as needed for KrInstance

    Returns:
        (DataFrame) chart_data: df with absolute chart positions of each planet, house (equal houses), MC/IC, and nodes

    """
    data = {'names': [], 'sun': [], 'moon': [], 'mercury': [], 'venus': [], 'mars': [], 'jupiter': [], 'saturn': [], 'uranus': [], 'neptune': [], 'pluto': [], 'mean_nn': [], 'true_nn': [], 
    'asc': [], 'second_house': [], 'third_house': [], 'fourth_house': [], 'fifth_house': [], 'sixth_house': [], 
    'seventh_house': [], 'eighth_house': [], 'ninth_house': [], 'tenth_house': [], 'eleventh_house': [], 'twelfth_house': [],
    'nadir': [], 'midheaven': []}
    for bd in birth_data.itertuples(index=False, name=None):
        try:
            chart = KrInstance(*bd)
        except:
            continue
        process_chart(chart, data)
    chart_data = pd.DataFrame(data)
    chart_data.set_index('names')
    return chart_data


def process_chart(chart: KrInstance, data):
    """
    Helper: processes the passed in chart (as a KrInstance) to update the 
    data dict passed in, appending to each list appropriately. 

    Args: 
        (KrInstance) chart: chart in question
        (dict) data: dict of planet/house positions to list of absolute positions for those bodies

    """
    iterplanets = list(data.keys())
    data['names'].append(chart.name)
    planets = chart.planets_degrees
    planet = 1
    for i in range(len(planets)):
        data[iterplanets[planet]].append(planets[i])
        planet += 1
    curr_house = chart.first_house['abs_pos']
    for j in range(NUM_HOUSES):   # populate houses w equal house system
        data[iterplanets[planet]].append(curr_house)
        curr_house = (curr_house + SIGN_ARC) % 360
        planet += 1
    # since KrInstance uses Placidus, populate nadir and midheaven w fourth and tenth house
    data['nadir'].append(chart.fourth_house['abs_pos'])
    data['midheaven'].append(chart.tenth_house['abs_pos'])


def parse_birth_data(filename):
    """ 
    Parse the AstroDataBank for the birth data of all entries, 
    and return the birth data in appropriate form for read_chart 
    as pandas dataframe. 

    Args:
        filename: filepath of astrodatabank

    Returns: 
        (DataFrame) DataFrame of Birth Data
    """
    iterparser = {"public_data": ["sflname", "roddenrating", "iyear", "imonth", "iday",
                                  "sbtime", "place", "sctr", "slong", "slati", "time_unknown"]}
    colnames = ['name', 'roddenrating', 'year', 'month', 'day', 'time',
                'city', 'country', 'lon', 'lat', 'time_unknown']
    def convert(tude: str): 
        neg = -1 if 's' in tude or 'w' in tude else 1
        letter = tude.find(next(filter(str.isalpha, tude)))
        return neg * (float(tude[:letter]) + float(tude[letter+1:]) / 600)
    
    birth_data = pd.read_xml(filename, parser='lxml', iterparse=iterparser, converters={'slong': convert, 'slati': convert})
    birth_data.rename(columns={ curr_name:desired_name for curr_name, desired_name in zip(iterparser['public_data'], colnames) }, inplace=True)
    
    # clean up: exclude Rodden Ratings lower than A and unknown birth times
    bad_rating = birth_data[((birth_data['roddenrating'] != 'A') & (birth_data['roddenrating'] != 'AA')) | (birth_data['time_unknown'] == 'yes')].index
    birth_data.drop(index=bad_rating, inplace=True)
    
    # split birth time into hour and min
    birth_data['hr'] = birth_data.apply(lambda series: int(series['time'].split(':')[0]), axis=1)
    birth_data['min'] = birth_data.apply(lambda series: int(series['time'].split(':')[1]), axis=1)

    # clean up names so things can be passed to kerykeion easily
    birth_data = birth_data[['name', 'year', 'month', 'day', 'hr', 'min', 'city', 'country', 'lon', 'lat']]
    birth_data.reset_index(drop=True, inplace=True)
    
    # finally, get timezone
    tf = TimezoneFinder()
    birth_data['tz'] = [tf.timezone_at(lng=birth_data['lon'][i], lat=birth_data['lat'][i]) for i in range(birth_data.shape[0])]
    return birth_data


def scrape_bio_data(link):
    """
    Given a link to the AstroDataBank wiki page for a given individual, scrape the page for their biography
    and return the biography as a string.

    Args:
        (str) link: link to a page on AstroDataBank wiki
    Returns:
        (str) bio: biography as one string
    """
    bio = ''
    r = requests.get(link, timeout=5)
    r.raise_for_status()
    page = r.text
    tree = BeautifulSoup(page, 'lxml')
    # Since ADB's HTML is badly labeled/nested, we have to use something of a clunky for loop.
    head = tree.find('span', id='Biography').parent
    for paragraph in head.next_siblings:
        if type(paragraph) is not element.Tag: continue
        if len(paragraph.contents) > 1 or paragraph.name != 'p': break
        bio += str(paragraph.string).strip()
        bio += ' '
    print(bio)
    return bio.strip()



def parse_life_data(filename, birth_data):
    """
    Parse the AstroDataBank for the life data stored on the wiki, store it in a passed in DataFrame
    of birth_data.

    Args:
        (str) filename: filepath of the AstroDataBank in XML
        (DataFrame) birth_data: for merging with new frame of life data, indexed by name 
    Returns:
        (DataFrame) life_data: indexed by name, DataFrame with strings of natural language biographies
    """
    iterparser = {'adb_entry': ['sflname', 'adb_link']}
    life_data = pd.read_xml(filename, parser='lxml', iterparse=iterparser)
    life_data.rename(columns={'sflname':'names', 'adb_link': 'link'}, inplace=True)
    life_data.set_index('names', inplace=True)
    life_data = pd.merge(birth_data, life_data, left_index=True, right_index=True)
    life_data['bios'] = life_data.apply(lambda series: scrape_bio_data(series['link']), axis=1)
    life_data.drop(columns='link', inplace=True)
    return life_data


def generate_random_charts(n):
    """
    Use Kerykeion to generate random charts with random birth data, generated
    just with numpy's random seeding.
    Args:
        (int) n: the number of desired charts
    Returns: 
        (DataFrame) chart_data: chart_data generated by Kerykeion
    """
    # we have to instance kerykeion with names: just use numbers
    names = [f'random_chart {i}' for i in range(n)]
    random_datetimes = [gen_random_datetime.gen_datetime(min_year=1700) for _ in range(n)]
    random_longs = [180. * np.random.random_sample() - 90. for _ in range(n)]
    random_lats = [360. * np.random.random_sample() - 180. for _ in range(n)]
    tf = TimezoneFinder()
    random_tzs = [tf.timezone_at(lng = long, lat = lati) for long, lati in zip(random_longs, random_lats)]
    birth_data = pd.DataFrame({
        'names': names,
        'year': [dt.year for dt in random_datetimes],
        'month': [dt.month for dt in random_datetimes],
        'day': [dt.day for dt in random_datetimes],
        'hr': [dt.hour for dt in random_datetimes],
        'min': [dt.minute for dt in random_datetimes],
        'lon': random_longs,
        'lat': random_lats,
        'tz': random_tzs
    })
    return read_charts(birth_data)


def generate_random_bios(n):
    """
    Use GPT-3 to generate random biographies, n times.
    Args:
        (int) n: number of bios to generate
    Returns:
        (DataFrame) bios: biographies stored in DataFrame
    """
    def prompt_GPT_3(gpt_prompt):
        """
        A little helper with pre-calibrated settings for giving and getting one answer from
        GPT-3 at a time.
        Args:
            (str) gpt_prompt: prompting question
        Returns:
            (str) answer: GPT's response
        """
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=gpt_prompt,
            temperature=0.75,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response['choices'][0]['text']
    return pd.DataFrame({
        'bios': [prompt_GPT_3('Give me a random biography of a human person.') for _ in range(n)]
    })


def conglomerate_bios_and_charts(chart_data: pd.DataFrame, bios: pd.DataFrame):
    """
    A simple helper, just concatenates chart data and bios and returns it!
    Args:
        (DataFrames): chart_data is a DF of chart data, bios is a DF (really a Series) of bios!
    """
    return pd.concat((chart_data, bios), axis=1)



def shuffle_bios_and_charts(chart_data: pd.DataFrame):
    """
    Generate perturbed examples from array of examples with bios, 
    i.e. examples where biographies are matched with incorrect birth times. 
    Args:
        (DataFrame) chart_data: chart data WITH biographies, perhaps correct perhaps not
    Returns:
        (DataFrame) bad_chart_data: chart data with biographies totally scrambled
    """
    bad_chart_data = chart_data.copy(deep=True)
    still_good = chart_data['bios'] == bad_chart_data['bios']
    while not bad_chart_data[still_good].empty:   # reshuffle till there's no matches anymore
        bad_chart_data.loc[still_good, 'bios'] = chart_data['bios'].sample(n=bad_chart_data[still_good].shape[0], replace=True).values
        still_good = chart_data['bios'] == bad_chart_data['bios']    # update the mask
    bad_chart_data['labels'] = 0
    return bad_chart_data


def concatenate_and_split(*data):
    """
    Concatenate all data in passed in tuple, then split into train and test sets!
    Also ignores however the dataFrames passed in are indexed. For use before split_x_y.

    Args:
        (tuple) data: tuple of DataFrames, with labels!
    Returns: 
        (tuple) train_set, test_set: training and test sets with mix of both good and bad data in each
    """
    big_frame = pd.concat(data, ignore_index=True)
    return train_test_split(big_frame, test_size=0.2)



def split_x_y(*data, names):
    """
    Helper method takes in possibly multiple dataframes, splits label column off the end
    and saves both into separate files.
    Args: 
        (tuple of DataFrames): dfs which have a labels column
        (sequence): sequence of names for the x-y split. Assumes same length as the tuple of dfs!
    Returns:
        None
    """
    name_idx = 0
    for df in data:
        X = df.drop(columns='labels')
        y = df['labels']
        X.to_csv(f'{names[name_idx]}_data.csv', index=False)
        y.to_csv(f'{names[name_idx]}_labels.csv', index=False)
        name_idx += 1


def extract_bios(data: pd.DataFrame):
    """
    Self-explanatory. Gets the bios from a DataFrame and returns them. For FunctionTransformer.
    Args:
        (DataFrame) data: df of chart data, with bios column
    Returns:
        (Series) bios: series of bios
    """
    return data['bios'].astype('str')

def extract_chart(data: pd.DataFrame):
    """
    Self-explanatory. Gets the numerical chart data from a DF and returns it. For FunctionTransformer.
    Args:
        (DataFrame) data: df of chart data, with bios column
    Returns:
        (Series) bios: series of bios
    """
    return data.loc[:, ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto', 'asc', 'midheaven', 'true_nn']]



def load_csv_files(*filenames):
    """
    Helper function calls read_csv on passed in filenames and returns dfs.
    Args:
        (tuple of str) filenames: valid filepaths to csv files
        (kwargs) read_options: any keyword arguments to be passed to read_csv
    Returns:
        (tuple of DataFrame) dfs: Frames of read-in data
    """
    return tuple(pd.read_csv(filename) for filename in filenames)



def main():
    pass       


if __name__ == "__main__":
    main()