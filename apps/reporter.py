import jinja2
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from IPython.display import display


class Reporter:
    """
    Class for collecting and printing data in tabular form
    """

    def __init__(self, attribute_col: str = "Attribute", result_col: str = "Result"):
        self._border_sign = "#"
        self._hor_split_sign = "~"
        self._ver_split_sign = "|"
        self._tolerance = 4
        self._max_len_lf: int = 0
        self._max_len_rt: int = 0
        self._data_list: list[tuple[str, str]] = []
        self.attribute_col = attribute_col
        self.result_col = result_col

    @property
    def max_len_lf(self) -> int:
        """Maximum left column width (the 1st column)"""
        return self._max_len_lf

    @property
    def max_len_rt(self) -> int:
        """Maximum right column width (the 2nd column)"""
        return self._max_len_rt

    @property
    def data_list(self) -> list[tuple[str, str]]:
        """The list of data to be printed"""
        return self._data_list

    @property
    def border_sign(self) -> str:
        """Character used for the outside table border"""
        return self._border_sign

    @border_sign.setter
    def border_sign(self, value: str) -> None:
        self._border_sign = value

    @property
    def hor_split_sign(self) -> str:
        """Character used for horizontal splitter in tabular form"""
        return self._hor_split_sign

    @hor_split_sign.setter
    def hor_split_sign(self, value: str) -> None:
        self._hor_split_sign = value

    @property
    def ver_split_sign(self) -> str:
        """Character used for vertical splitter in tabular form"""
        return self._ver_split_sign

    @ver_split_sign.setter
    def ver_split_sign(self, value: str) -> None:
        self._ver_split_sign = value

    @property
    def tolerance(self) -> int:
        """Number of digits after the floating point"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: int) -> None:
        self._tolerance = value

    @property
    def attribute_col(self) -> str:
        """Name of the attribute (the 1st column)"""
        return self._attribute_col

    @attribute_col.setter
    def attribute_col(self, value: str) -> None:
        array_str_key = value.splitlines()
        for str_key in array_str_key:
            self._max_len_lf = max(self.max_len_lf, len(str_key))
        self._attribute_col = value

    @property
    def result_col(self) -> str:
        """Name of the result (the 2nd column)"""
        return self._result_col

    @result_col.setter
    def result_col(self, value: str) -> None:
        array_str_value = value.splitlines()
        for str_value in array_str_value:
            self._max_len_rt = max(self.max_len_rt, len(str_value))
        self._result_col = value

    def format_value(self, value: float) -> str:
        """
        Format value to string
        :param value: input value
        :return: string of the value
        """
        return f"{value:.{self.tolerance}f}"

    def format_matrix(self, matrix: np.ndarray) -> str:
        """
        Formatting matrix into string
        :param matrix: input matrix
        :return: string of the matrix
        """
        return np.array2string(matrix, precision=self.tolerance)

    @staticmethod
    def format_link(link: str) -> str:
        """
        Format link to string
        :param link: input link
        :return: formatted string of the link
        """
        return f'<a target="_blank" href="{link}">{link}</a>'

    def add_item(self, key: str, value: str) -> None:
        """
        Form the report data that consist a multi strings
        :param key: attribute
        :param value: result
        """
        array_str_key = key.splitlines()
        array_str_value = value.splitlines()

        for str_key in array_str_key:
            self._max_len_lf = max(self.max_len_lf, len(str_key))

        for str_value in array_str_value:
            self._max_len_rt = max(self.max_len_rt, len(str_value))

        self._data_list.append((key, value))

    def _print_line_splitter(self, sign: str = "-") -> None:
        """
        Print line splitter
        """
        print(sign * (self.max_len_lf + self.max_len_rt + 5))

    def _print_multi_string(self, row: tuple[str, str]) -> None:
        """
        Print the multi-string
        :param row: tuple of data
        """
        array_str_key = row[0].splitlines()
        array_str_value = row[1].splitlines()
        num_str = max(len(array_str_key), len(array_str_value))

        for index in range(num_str):
            if index < len(array_str_key):
                left_str = f'{f"{array_str_key[index]}":<{self.max_len_lf}}'
            else:
                left_str = f'{f" ":<{self.max_len_lf}}'

            if index < len(array_str_value):
                right_str = f'{f"{array_str_value[index]:<{self.max_len_rt}}"}'
            else:
                right_str = f'{f" ":<{self.max_len_rt}}'

            print(f" {left_str} {self.ver_split_sign} {right_str}")

    def refresh_table_config(self) -> None:
        """
        Refresh table configuration, like line lengths
        :return:
        """
        self._max_len_lf = 0
        self._max_len_rt = 0

        self.attribute_col = self.attribute_col
        self.result_col = self.result_col

        temp_data_list = self.data_list
        self._data_list = []
        for temp_data in temp_data_list:
            self.add_item(temp_data[0], temp_data[1])

    def print_report(self) -> None:
        """
        Print report data
        """
        self._print_line_splitter(self.border_sign)
        self._print_multi_string((self.attribute_col, self.result_col))
        self._print_line_splitter(self.border_sign)
        for item_data in self.data_list[0:-1]:
            self._print_multi_string(item_data)
            self._print_line_splitter("~")
        self._print_multi_string(self.data_list[-1])
        self._print_line_splitter(self.border_sign)

    def print_refreshed_report(self) -> None:
        """
        Print report data with a refreshing table
        """
        self.refresh_table_config()
        self.print_report()

    def get_pd_report(self, caption: str = "") -> Styler:
        """
        Generate report data using pandas dataframe
        :param caption: caption
        :return: styler
        """
        df_data_list = pd.DataFrame(self.data_list, columns=[self.attribute_col, self.result_col])
        pd.set_option('display.max_colwidth', None)
        df_data_list = (df_data_list.style.hide(axis='index')
        .set_properties(**{
            'text-align': 'left',
            'white-space': 'pre-wrap',
            'border': '1px solid lightgrey'
        }).set_table_styles([  # type: ignore
            {"selector": "th", "props": [("text-align", "left")]}
        ])).set_caption(caption)
        return df_data_list

    def print_pd_report(self, caption: str = "") -> None:
        """
        Print report data using pandas dataframe
        """
        df_data_list = self.get_pd_report(caption)
        display(df_data_list)

    @staticmethod
    def get_concat_pd_reports(dfs: list[tuple[str, Styler]], caption: str = "") -> Styler:
        """
        Generate report data using pandas dataframes
        :param dfs: a list of pandas dataframes
        :param caption: caption
        :return: styler
        """
        attribute_col_name, result_col_name = dfs[0][1].data.columns
        column_name_list = dfs[0][1].data[attribute_col_name].values.tolist()
        concat_dfs = pd.DataFrame(columns=column_name_list)

        for name, item in dfs:
            df_results = item.data[result_col_name].values.tolist()
            concat_dfs.loc[name] = df_results

        pd.set_option('display.max_colwidth', None)
        concat_dfs = (concat_dfs.style
        .set_properties(**{
            'text-align': 'left',
            'white-space': 'pre-wrap',
            'border': '1px solid lightgrey'
        }).set_table_styles([  # type: ignore
            {"selector": "th", "props": [("text-align", "left")]}
        ])).set_caption(caption)
        return concat_dfs

    @staticmethod
    def print_concat_pd_reports(dfs: list[tuple[str, Styler]], caption: str = "") -> None:
        """
        Print report data using pandas dataframes
        """
        concat_dfs = Reporter.get_concat_pd_reports(dfs, caption)
        display(concat_dfs)
