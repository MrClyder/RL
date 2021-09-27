import pandas as pd
from haversine import haversine

def cal_distance(row):
    """
    计算两个经纬度点之间的距离
    """
    long1 = row['long1']
    lat1 = row['lat1']
    long2 = row['long2']
    lat2 = row['lat2']
    g1 = (long1, lat1)
    g2 = (long2, lat2)

    ret = haversine(g1, g2) * 1000
    result = "%.7f" % ret

    return result

