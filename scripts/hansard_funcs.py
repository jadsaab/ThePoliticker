# Define functions for inspecting Hansard transcripts
# df refers to transcript(s) in pandas dataframe format
# ----------------------------------------------------------------------

def get_num_unique_speakers(df):
    """
    Function that prints the number of unique speakers
    """

    speakers = df['speakername'].unique()
    print('Number of unique speakers: %s' %len(speakers))


def inspect_row(df, row_num):
    """
    Function that prints specified row of dataframe
    """

    for row in row_num:
        print('\nInspecting row %d:\n%s\n' % (row, df.loc[row, :]))


def inspect_speaker(df, speaker):
    """
    Function that prints rows of the dataframe related to specified speaker
    """

    matches = df['speakername'].str.contains(speaker, regex=True)
    print('\nInspecting speaker %s:\n%s\n' % (speaker, df.loc[matches[matches].index, :]))


def show_names_containing(df, name):
    """
    Function that prints all speakers containing the specified name
    """

    matches = df['speakername'].str.contains(name, regex=True)
    print('\nUnique occurences of %s:\n%s\n' % (name, df['speakername'][matches[matches].index].unique()))


def overwrite_speaker(df, old_name, new_name, row_to_check):
    """
    Function that overwrites an existing name
    e.g. Mr. Smith -> John Smith
    """

    df['speakername'] = df['speakername'].replace('^'+old_name, new_name, regex=True)
    for row in row_to_check:
        print('Row after overwriting %s with %s:\n%s\n' % (old_name, new_name, df.loc[row,:]))


def get_key_from_value(dictionary, val):
    """
    Function that gets the dictionary key(s) associated with a value
    e.g. If a dictionary contains names of speakers, for a specified speaker,
    get every key whose value contains the speaker
    """

    all_keys = []
    for key, value in dictionary.items():
         if val == value:
             all_keys.append(key)
    return all_keys
