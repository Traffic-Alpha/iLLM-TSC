'''
Author: pangay 1623253042@qq.com
Date: 2024-03-27 14:54:18
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2024-06-26 01:08:30
FilePath: /TSC-HARLA/parse_trip_info.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tshub.utils.parse_trip_info import TripInfoStats
from tshub.utils.get_abs_path import get_abs_path

path_convert = get_abs_path(__file__)

if __name__ == '__main__':
    
    trip_info = path_convert(f'./Result/rule.tripinfo.xml')
    stats = TripInfoStats(trip_info)
    stats.to_csv(path_convert(f'./Result/rule.csv'))