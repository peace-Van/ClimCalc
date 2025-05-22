"""
This file contains the definition of the climate classification methods

Author: Tracy Van
Date: 2024-12-08
"""

import numpy as np

EPS = 1e-5

class KoppenClassification:
    order = [
        "Af",
        "Am",
        "Aw",
        "BWh",
        "BWk",
        "BSh",
        "BSk",
        "Cfa",
        "Cfb",
        "Cfc",
        "Csa",
        "Csb",
        "Csc",
        "Cwa",
        "Cwb",
        "Cwc",
        "Dfa",
        "Dfb",
        "Dfc",
        "Dfd",
        "Dsa",
        "Dsb",
        "Dsc",
        "Dsd",
        "Dwa",
        "Dwb",
        "Dwc",
        "Dwd",
        "ET",
        "EF",
    ]

    color_map = {
        # A组 - 热带气候
        "Af": "#960000",  # 热带雨林
        "Am": "#FF0000",  # 热带季风
        "Aw": "#FF6666",  # 热带草原
        # B组 - 干燥气候
        "BWh": "#FF9933",  # 热带沙漠
        "BWk": "#FFCC00",  # 温带沙漠
        "BSh": "#CC9900",  # 热带草原
        "BSk": "#CC6600",  # 温带草原
        # C组 - 温带气候
        "Cfa": "#33CC33",  # 温带湿润
        "Cfb": "#009900",  # 海洋性
        "Cfc": "#006600",  # 亚寒带海洋性
        "Csa": "#66FF66",  # 地中海夏干
        "Csb": "#00FF00",  # 温和地中海
        "Csc": "#004D00",  # 亚寒带地中海
        "Cwa": "#99FF99",  # 亚热带季风
        "Cwb": "#CCFFCC",  # 高地热带
        "Cwc": "#669966",  # 亚寒带季风
        # D组 - 大陆性气候
        "Dfa": "#00CCFF",  # 温带大陆性
        "Dfb": "#0099CC",  # 亚寒带大陆性
        "Dfc": "#006699",  # 亚寒带
        "Dfd": "#003366",  # 极寒带大陆性
        "Dsa": "#99CCFF",  # 地中海型大陆性
        "Dsb": "#6699CC",  # 温和地中海型大陆性
        "Dsc": "#336699",  # 亚寒带地中海型大陆性
        "Dsd": "#1A334D",  # 极寒带地中海型大陆性
        "Dwa": "#CCE5FF",  # 季风型大陆性
        "Dwb": "#3366CC",  # 温和季风型大陆性
        "Dwc": "#0033CC",  # 亚寒带季风型大陆性
        "Dwd": "#000066",  # 极寒带季风型大陆性
        # E组 - 极地气候
        "ET": "#99FFFF",  # 苔原
        "EF": "#FFFFFF",  # 冰原
    }

    @staticmethod
    def classify(climate_data: np.ndarray, cd_threshold: float, kh_mode: str) -> str:
        """
        cd_threshold: threshold for C and D classes, usually -3 or 0
        kh_mode: classification mode for B classes ('coldest_month' coldest month < cd_threshold is k, otherwise h, or 'mean_temp' mean temperature < 18 is k, otherwise h)
        """
        # Sort by temperature
        t_monthly = climate_data[:, np.argsort(climate_data[0, :])]
        # print(t_monthly)

        # Calculate basic climate indicators
        total_pr = np.sum(t_monthly[1, :])
        pr_percent = np.sum(t_monthly[1, 5:12]) / (
            total_pr + EPS
        )  # Summer half-year precipitation ratio
        mean_temp = np.mean(t_monthly[0, :])

        # Calculate aridity threshold
        thresh = 20 * mean_temp
        if pr_percent >= 0.7:
            thresh += 280
        elif pr_percent >= 0.3:
            thresh += 140

        cls = ""

        # Determine climate type
        if total_pr < 0.5 * thresh:  # Desert climate BW
            cls = "BW"
            if kh_mode == "coldest_month":
                if t_monthly[0, 0] < cd_threshold:
                    cls += "k"
                else:
                    cls += "h"
            elif kh_mode == "mean_temp":
                if mean_temp < 18:
                    cls += "k"
                else:
                    cls += "h"

        elif total_pr < thresh:  # Steppe climate BS
            cls = "BS"
            if kh_mode == "coldest_month":
                if t_monthly[0, 0] < cd_threshold:
                    cls += "k"
                else:
                    cls += "h"
            elif kh_mode == "mean_temp":
                if mean_temp < 18:
                    cls += "k"
                else:
                    cls += "h"

        else:
            if t_monthly[0, 0] >= 18:  # Tropical climate A
                cls = "A"
                if np.min(t_monthly[1, :]) >= 60:
                    cls += "f"
                else:
                    thresh_2 = 100 - total_pr / 25
                    if np.min(t_monthly[1, :]) < thresh_2:
                        cls += "w"
                    else:
                        cls += "m"

            elif t_monthly[0, -1] < 10:  # Polar climate E
                cls = "E"
                if t_monthly[0, -1] < 0:
                    cls += "F"
                else:
                    cls += "T"

            elif t_monthly[0, 0] > cd_threshold:  # Temperate climate C
                cls = "C"
                # Determine precipitation characteristics
                if np.max(t_monthly[1, 9:12]) > 10 * np.min(t_monthly[1, 0:3]):
                    cls += "w"
                elif (
                    np.max(t_monthly[1, 0:3]) > 3 * np.min(t_monthly[1, 9:12])
                    and np.min(t_monthly[1, 9:12]) < 30
                ):
                    cls += "s"
                else:
                    cls += "f"

                # Determine temperature characteristics
                if np.sum(t_monthly[0, :] > 10) < 4:
                    cls += "c"
                elif t_monthly[0, -1] > 22:
                    cls += "a"
                else:
                    cls += "b"

            else:  # Continental climate D
                cls = "D"
                # Determine precipitation characteristics
                if np.max(t_monthly[1, 9:12]) > 10 * np.min(t_monthly[1, 0:3]):
                    cls += "w"
                elif (
                    np.max(t_monthly[1, 0:3]) > 3 * np.min(t_monthly[1, 9:12])
                    and np.min(t_monthly[1, 9:12]) < 30
                ):
                    cls += "s"
                else:
                    cls += "f"

                # Determine temperature characteristics
                if np.sum(t_monthly[0, :] > 10) < 4:
                    if t_monthly[0, 0] <= -38:
                        cls += "d"
                    else:
                        cls += "c"
                elif t_monthly[0, -1] > 22:
                    cls += "a"
                else:
                    cls += "b"

        return cls


class TrewarthaClassification:
    order = ["Ar", "Aw", "BW", "BS", "Cf", "Cs", "Do", "Dc", "Eo", "Ec", "Ft", "Fi"]

    color_map = {
        # A组 - 热带气候
        "Ar": "#960000",  # 热带雨林 (深红)
        "Aw": "#FF6666",  # 热带草原 (浅红)
        # B组 - 干燥气候
        "BW": "#FF9933",  # 沙漠 (橙色)
        "BS": "#CC9900",  # 草原 (金色)
        # C组 - 副热带气候
        "Cf": "#33CC33",  # 湿润副热带 (深绿)
        "Cs": "#99FF99",  # 地中海副热带 (浅绿)
        # D组 - 温带气候
        "Do": "#00CCFF",  # 海洋性温带 (浅蓝)
        "Dc": "#0066CC",  # 大陆性温带 (深蓝)
        # E组 - 亚寒带气候
        "Eo": "#6666FF",  # 海洋性亚寒带 (浅紫)
        "Ec": "#333399",  # 大陆性亚寒带 (深紫)
        # F组 - 极地气候
        "Ft": "#99FFFF",  # 苔原 (浅青)
        "Fi": "#FFFFFF",  # 冰原 (白色)
    }

    @staticmethod
    def classify(climate_data: np.ndarray) -> str:
        # Sort by temperature
        t_monthly = climate_data[:, np.argsort(climate_data[0, :])]

        # Calculate basic climate indicators
        total_pr = np.sum(t_monthly[1, :])
        pr_percent = np.sum(t_monthly[1, 0:6]) / (
            total_pr + EPS
        )  # Winter half-year precipitation ratio
        mean_temp = np.mean(t_monthly[0, :])

        # Calculate aridity threshold (Trewartha's formula)
        # de Castro M, Gallardo C, Jylha K, Tuomenvirta H (2007) The use of a climate-type classification for assessing climate change effects in Europe form an ensemble of nine regional climate models. Clim Change 81: 329−341
        thresh = 23 * mean_temp - 6.4 * pr_percent + 410

        cls = ""

        # Determine climate type
        if total_pr < 0.5 * thresh:  # Desert climate BW
            cls = "BW"

        elif total_pr < thresh:  # Steppe climate BS
            cls = "BS"

        else:
            if t_monthly[0, 0] >= 18:  # Tropical climate A
                cls = "A"
                # Determine precipitation characteristics
                if (
                    np.sum(t_monthly[1, :] >= 60) >= 10
                ):  # At least 10 months with precipitation >= 60mm
                    cls += "r"
                else:
                    cls += "w"

            elif t_monthly[0, -1] < 10:  # Polar climate F
                cls = "F"
                if t_monthly[0, -1] < 0:  # Warmest month temperature < 0℃
                    cls += "i"
                else:  # Warmest month temperature 0-10℃
                    cls += "t"

            elif np.sum(t_monthly[0, :] > 10) >= 8:  # Subtropical climate C
                cls = "C"
                # Determine precipitation characteristics
                if (
                    np.max(t_monthly[1, 0:3]) > 3 * np.min(t_monthly[1, 9:12])
                    and np.min(t_monthly[1, 9:12]) < 30
                    and total_pr < 890
                ):
                    cls += "s"
                else:
                    cls += "f"

            elif np.sum(t_monthly[0, :] > 10) >= 4:  # Temperate climate D
                cls = "D"
                if t_monthly[0, 0] < 0:  # Coldest month < 0℃
                    cls += "c"
                else:  # Coldest month >= 0℃
                    cls += "o"

            else:  # Subarctic climate E
                cls = "E"
                if t_monthly[0, 0] < -10:  # Coldest month < -10℃
                    cls += "c"
                else:  # Coldest month >= -10℃
                    cls += "o"

        return cls


class DLClassification:
    order = [
        "Af",
        "Am",
        "Aw",
        "Bm",
        "Bw",
        "Bx",
        "Cf",
        "Cm",
        "Cw",
        "Cs",
        "Df",
        "Dm",
        "Dw",
        "Ds",
        "Dx",
        "Em",
        "Ew",
        "Es",
        "Ex",
        "Ff",
        "Fm",
        "Fx",
        "Gm",
        "Gw",
        "Gs",
        "Gx",
    ]

    class_map = [
        "Dm",
        "Ds",
        "Df",
        "Fx",
        "Cf",
        "Em",
        "Gw",
        "Gm",
        "Cw",
        "Dx",
        "Ex",
        "Aw",
        "Bm",
        "Bx",
        "Fm",
        "Ff",
        "Cm",
        "Dw",
        "Af",
        "Cs",
        "Es",
        "Ew",
        "Bw",
        "Gx",
        "Gs",
        "Am",
    ]

    color_map = {
        # A组 - 热带湿润气候
        "Af": "#FF0000",  # 深红色，热带雨林
        "Am": "#FF6666",  # 浅红色，热带季风
        "Aw": "#FFD580",  # 浅橙黄，热带草原（优化后）
        
        # B组 - 干燥气候
        "Bm": "#CC3300",  # 浅橙色，温和沙漠
        "Bw": "#FF9933",  # 亮橙色，季节性半干旱（更亮）
        "Bx": "#993300",  # 深棕色，极端沙漠
        
        # C组 - 亚热带湿润气候
        "Cf": "#006600",  # 深绿色，亚热带海洋性
        "Cm": "#99FF99",  # 浅绿色，亚热带季风
        "Cw": "#CCFF99",  # 黄绿色，干冬亚热带
        "Cs": "#66CC00",  # 黄绿色，地中海气候
        
        # D组 - 温带气候
        "Df": "#00CC99",  # 青绿色, 温带海洋性（偏绿）
        "Dm": "#009999",  # 中蓝色，湿润大陆性
        "Dw": "#99CCFF",  # 浅蓝色，干冬大陆性
        "Ds": "#007CC0",  # 青蓝色，半湿润大陆性（偏绿）
        "Dx": "#D9C97C",  # 灰黄色，干旱温带（与Bm/Bx/Ex区分）
        
        # E组 - 高地和干旱大陆性气候
        "Em": "#9966CC",  # 紫色，温和高地
        "Ew": "#FFCC00",  # 金黄色，干冬半干旱
        "Es": "#FFCC99",  # 肉色，干夏半干旱
        "Ex": "#A67C29",  # 深土黄色，极端干旱（优化后）
        
        # F组 - 亚极地和亚寒带气候
        "Ff": "#3366CC",  # 亮蓝色，亚极地海洋性
        "Fm": "#006699",  # 深蓝紫色，亚寒带大陆性
        "Fx": "#00CCCC",  # 深青色，极端大陆性
        
        # G组 - 极地气候
        "Gm": "#000099",  # 深蓝色，亚极地-极地过渡
        "Gw": "#CC99FF",  # 浅紫色，高山苔原
        "Gs": "#99FFFF",  # 浅青色，凉爽夏季极地
        "Gx": "#FFFFFF"   # 白色，寒冷夏季极地
    }

    # bulk operation
    @classmethod
    def classify(cls, probabilities: np.ndarray) -> list[str]:
        cls_indices = np.argmax(probabilities, axis=1)
        return [cls.class_map[i] for i in cls_indices]


# # usage
# if __name__ == "__main__":
#     from DLModel import DenseNetwork
#     import h5py

#     np.set_printoptions(suppress=True, precision=4)

#     example_data = np.array(
#         [
#             [
#                 [
#                     2.7,
#                     4.9,
#                     8.4,
#                     12.9,
#                     17.2,
#                     20.5,
#                     22.1,
#                     21.7,
#                     18.9,
#                     14.7,
#                     9.6,
#                     4.2,
#                 ],  # low temperature in Celsius
#                 [
#                     8.1,
#                     11.4,
#                     24.1,
#                     44.9,
#                     78.0,
#                     109.5,
#                     231.8,
#                     217.1,
#                     120.8,
#                     42.6,
#                     14.8,
#                     6.2,
#                 ],  # precipitation in mm
#                 [
#                     9.3,
#                     12.1,
#                     16.8,
#                     22.5,
#                     26.3,
#                     28.3,
#                     30.0,
#                     29.9,
#                     25.7,
#                     20.7,
#                     16.0,
#                     10.7,
#                 ],  # high temperature in Celsius
#             ]
#         ], dtype=np.float32)
#     # shape should be (batch_size, 3, 12)
#     print("Input Shape:", example_data.shape)

#     # data may also be loaded from the downloaded data file
#     # with h5py.File('climate_data_land.h5', 'r') as f:
#     #     indices = f.get('indices')[:]       # latitude and longitude of all the locations
#     #     # example is given as a single location and single year, you can also do bulk processing
#     #     lat, lon = 30.6, 105.8
#     #     year = 2010
#     #     # round to the nearest 0.5 degree
#     #     lat = round(lat * 2) / 2
#     #     lon = round(lon * 2) / 2
#     #     idx = np.where((indices[:, 0] == lat) & (indices[:, 1] == lon))[0]
#     #     tmp = f.get('tmp')[idx, year - 1901, :]
#     #     pre = f.get('pre')[idx, year - 1901, :]
#     #     pet = f.get('pet')[idx, year - 1901, :]
#     #     example_data = np.array([tmp, pre, pet])
#     #     example_data = example_data.transpose(1, 0, 2)

#     with h5py.File("weights.h5", "r") as f:
#         network = DenseNetwork(f)
#         # pca_features, probability = network(example_data)
#         thermal_index, aridity_index, probability, veg_index = network(example_data)
#         advanced_dl_classifier = DLClassification(
#             order=DETAILED_ORDER,
#             class_map=DETAILED_MAP,
#             color_map=DETAILED_COLOR_MAP,
#         )
#         # basic_dl_classifier = DLClassification(
#         #     order=SIMPLE_ORDER,
#         #     class_map=SIMPLE_MAP,
#         #     color_map=SIMPLE_COLOR_MAP,
#         # )

#         # print("Cryohumidity:", pca_features[0][0])
#         # print("Continentality:", pca_features[1][0])
#         # print("Seasonality:", pca_features[2][0])
#         print("Thermal Index:", thermal_index[0])
#         print("Aridity Index:", aridity_index[0])
#         print("Probability:", probability)

#         print("DeepEcoClimate Type:", advanced_dl_classifier.classify(probability)[0])
#         # print("Basic DECC Type:", basic_dl_classifier.classify(probability)[0])
#         print("Land Cover Type:", VEG_MAP[veg_index[0]])

#         # For the classic climate classification, bulk operation is not supported
#         # example_data = example_data.squeeze()
#         # print(
#         #     "Köppen-Geiger Type:",
#         #     KoppenClassification.classify(
#         #         example_data, cd_threshold=-3, kh_mode="mean_temp"
#         #     ),
#         # )
#         # print("Trewartha Type:", TrewarthaClassification.classify(example_data))
