import pandas as pd


def split_column(df, col_name, new_col_names, separators):
    '''
      Simple function to split a column into multiple columns, given a few separators.

      Example:
      - df['X'] contains data: "aaaa, bbb, cc-dd"
      - split into four separate columns:
          separators    = [', ', ', ', '-']  # each separator is used only once
          new_names     = ['A', 'B', 'C', 'D']  # nr of columns should be nr of sepators + 1
          split_column(df, 'X', new_names, separators)

      input:
      - df: dataframe
      - col_name: name of the column to split
      - new_col_names: array with names of the new columns resulting from the split
      - separators: array with strings of separators to split on
      returns:
      - dataframe with the new columns

    '''

    # initialize index
    idx = 0

    # set the active column, this column will be evaluated for splitting
    active_column = df[col_name]

    # loop through separators
    for s in separators:
        # split active column on separator
        splitted_string = active_column.str.split(s)

        # make first part of the splitted a new column
        df[new_col_names[idx]] = splitted_string.str[0].astype(str, errors='raise')

        # make the remaining part of the split the new active column
        active_column = splitted_string.str[-1].astype(str, errors='raise')

        # update count
        idx += 1

    # final split
    df[new_col_names[idx]] = active_column
    return df


def convert_categories(df):
    df['Sex'] = pd.Categorical(df['Sex']).codes
    df['Embarked'] = pd.Categorical(df['Embarked']).codes
    df['Title'] = pd.Categorical(df['Title']).codes
    return df
