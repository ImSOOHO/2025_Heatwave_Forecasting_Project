%config InlineBackend.figure_format = "retina"
import warnings
warnings.filterwarnings(action='ignore') #경고 메시지 무시
from IPython.display import display #print가 아닌 display()로 연속 출력
from IPython.display import HTML #출력 결과를 HTML로 생성

import pandas as pd
from collections import OrderedDict
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import shapely
from shapely.geometry import Point
import sys
from pyproj import Proj, transform
%matplotlib inline
import folium
import os
import geopandas as gpd

!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')
