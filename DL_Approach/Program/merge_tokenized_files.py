import os

positive_database_path = 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\ExtractedData\\PositiveSamples'
negative_database_path = 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\ExtractedData\\NegativeSamples'

tokinized_positive_database_path = 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\ExtractedData\\TokenizedPositiveSamples'
tokinized_negative_database_path = 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\ExtractedData\\TokenizedNegativeSamples'


def merge_positive_tokenized_files():
    dirlist = os.listdir(positive_database_path)
    file_counter = 1
    out_file = os.path.join(tokinized_positive_database_path, "tokenized" + str(file_counter) + ".txt")
    file2 = open(out_file, "w")
    for file in dirlist:
        try:
            file1_path = positive_database_path + '\\' + file
            file1 = open(file1_path, "r")
            line = file1.read()
            file2.write(line+'\n')

            if os.path.getsize(out_file) > 30720000:  # 30 mb
                file2.close()
                file_counter += 1
                out_file = os.path.join(tokinized_positive_database_path, "tokenized" + str(file_counter) + ".txt")
                file2 = open(out_file, "w")

        except Exception as e:
            print("Error: Copying data ", file, e)


def merge_negative_tokenized_files():
    dirlist = os.listdir(negative_database_path)
    file_counter = 1
    out_file = os.path.join(tokinized_negative_database_path, "tokenized" + str(file_counter) + ".txt")
    file2 = open(out_file, "w")
    for file in dirlist:
        try:
            file1_path = negative_database_path + '\\' + file
            file1 = open(file1_path, "r")
            line = file1.read()
            file2.write(line + '\n')

            if os.path.getsize(out_file) > 30720000:  # 30 mb
                file2.close()
                file_counter += 1
                out_file = os.path.join(tokinized_negative_database_path, "tokenized" + str(file_counter) + ".txt")
                file2 = open(out_file, "w")

        except Exception as e:
            print("Error: Copying data ", file, e)



def merge_tokenized_files():
    merge_positive_tokenized_files()
    merge_negative_tokenized_files()
    print("Merging Files is done")
