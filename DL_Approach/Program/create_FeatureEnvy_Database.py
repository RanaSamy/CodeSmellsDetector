import os
import shutil
import pandas as pd
import logging

from DataModel.Smell import Smell
from DataModel.SmellModel import SmellModel

logging.basicConfig(
    filename='D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\ExtractedData\\log.log',
    filemode='w', level=logging.DEBUG)

positive_database_path = 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\ExtractedData\\PositiveSamples'
negative_database_path = 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\ExtractedData\\NegativeSamples'

tokenized_samples_path = 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Data'
data_metrics_root = 'D:\\Master\\Thesis\\RDT\\C_Data\\m_r'
smell_name = 'Feature Envy'

positive_list = set()
negative_list = set()


# The tool detected a instance of this smell because stdCharacters is more interested in members of the type: EjbInfo
def prepare_files_dict(metrics_df):
    files_dict = {}
    print("Preparing files Dict...")
    for index, metrics in metrics_df.iterrows():
        key = str(metrics['Package Name']) + "_" + str(metrics['Type Name'])
        if key not in files_dict:
            files_dict[key] = metrics['File path']
    return files_dict


def prepare_smells_dict(smells_df):
    smells_dict = {}
    print("Preparing Smells Dict...")
    for index, smell in smells_df.iterrows():
        key = str(smell['Package Name']) + "_" + str(smell['Type Name'])
        if key in smells_dict:
            smell_model = smells_dict.get(key)
            smell_model.smells_list.append(Smell(smell['Design Smell'], smell['Cause of the Smell']))
            is_smelly = True if smell['Design Smell'] == smell_name else False
            smell_model.is_smelly = smell_model.is_smelly or is_smelly
            logging.info('we have more than one smell for the same class: ' + key)
        else:
            smell_model = get_smell_model(smell)
            smells_dict[key] = smell_model
    return smells_dict


def get_smell_model(smell):
    smells_list = []
    smell_object = Smell(smell['Design Smell'], smell['Cause of the Smell'])
    smells_list.append(smell_object)
    is_smelly = True if smell['Design Smell'] == smell_name else False
    smell_model = SmellModel(is_smelly, smells_list)

    return smell_model


def is_FeatureEnvy_smell(cause_of_smell: str):
    if "instance" in cause_of_smell and "more interested in members of the type" in cause_of_smell:
        return True


def append_positive_negative_lists(smells_dict, files_dict):
    for smell_key in smells_dict:
        smell_value = smells_dict[smell_key]
        file_value = files_dict[smell_key]

        if smell_value.is_smelly:
            for smell in smell_value.smells_list:
                if smell.imp_smell == smell_name and is_FeatureEnvy_smell(
                        smell.cause_of_smell) and file_value not in positive_list:
                    positive_list.add(file_value)
        else:
            if file_value not in negative_list:
                negative_list.add(file_value)

    negative_list.difference_update(positive_list)


def move_files():
    dirlist = [item for item in os.listdir(tokenized_samples_path) if
               os.path.isdir(os.path.join(tokenized_samples_path, item))]
    for folder in dirlist:
        try:
            out_dir = os.path.join(tokenized_samples_path, folder)
            mapping_path = out_dir + '\Mapping.xlsx'
            dfs = pd.read_excel(mapping_path, sheet_name="sheet")
            for index, row in dfs.iterrows():
                if row[0] in positive_list:
                    shutil.copy2(row[1], positive_database_path)
                else:
                    shutil.copy2(row[1], negative_database_path)

        except Exception as e:
            print("Error: Something went wrong for moving files ", folder, e)


def create_feature_envy_dataset():
    dirlist = [item for item in os.listdir(data_metrics_root) if os.path.isdir(os.path.join(data_metrics_root, item))]
    for folder in dirlist:

        try:
            out_dir = os.path.join(data_metrics_root, folder)

            imp_smells_path = out_dir + '\DesignSmells.csv'
            method_metrics_path = out_dir + '\TypeMetrics.csv'

            smells_data = pd.read_csv(imp_smells_path, encoding='cp1252')
            smells_df = pd.DataFrame(smells_data, columns=['Project Name', 'Package Name', 'Type Name', 'Design Smell',
                                                           'Cause of the Smell'])

            metrics_data = pd.read_csv(method_metrics_path, encoding='cp1252')
            metrics_df = pd.DataFrame(metrics_data, columns=['Project Name', 'Package Name', 'Type Name', 'File path'])

            metrics_dict = prepare_files_dict(metrics_df)
            smells_dict = prepare_smells_dict(smells_df)
            append_positive_negative_lists(smells_dict, metrics_dict)

            print(folder, " is finished")
            print("------------------------------------------------------------------------")
            logging.info(folder + ' is finished')
            logging.info('------------------------------------------------------------------------')

        except Exception as e:
            print("Error: Something went wrong for ", folder, e)
            print(folder, " is finished with Exception")
            print("------------------------------------------------------------------------")

    move_files()
    print("Copying Files is done")
