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
        "As",
        "Bm",
        "Bw",
        "Bs",
        "Bx",
        "Cf",
        "Cm",
        "Cw",
        "Cs",
        "Df",
        "Dm",
        "Dw",
        "Ds",
        "Em",
        "Ew",
        "Es",
        "Ex",
        "Ff",
        "Fm",
        "Fw",
        "Fs",
        "Gm",
        "Gs",
        "Gx",
    ]

    class_map = [
        "Bw",
        "Bm",
        "Es",
        "Fw",
        "Em",
        "Ew",
        "Fm",
        "Cs",
        "Ff",
        "Fs",
        "Am",
        "Cf",
        "Df",
        "Ds",
        "Bx",
        "Bs",
        "Gm",
        "Aw",
        "Cm",
        "Dm",
        "Cw",
        "Gs",
        "As",
        "Ex",
        "Gx",
        "Af",
        "Dw",
    ]

    color_map = {
        # A组 - 热带湿润气候
        "Af": "#FF0000",  # 深红色，热带雨林
        "Am": "#FF6666",  # 浅红色，热带季风
        "Aw": "#FF9933",  # 浅橙黄，热带草原
        "As": "#CC6A00",  # 橙棕色，季节性半干旱
        
        # B组 - 干燥气候
        "Bm": "#FFD580",  # 浅橙色，温和沙漠
        "Bw": "#CC3300",  # 亮橙色，亚热带半沙漠
        "Bs": "#DAA520",  # 灰黄色，地中海半沙漠
        "Bx": "#993300",  # 深棕色，极端热沙漠
        
        # C组 - 亚热带湿润气候
        "Cf": "#006600",  # 深绿色，亚热带湿润
        "Cm": "#99FF99",  # 浅绿色，暖温带湿润
        "Cw": "#CCFF99",  # 黄绿色，东亚季风
        "Cs": "#66CC00",  # 黄绿色，地中海气候
        
        # D组 - 温带气候
        "Df": "#00CC99",  # 青绿色, 温带海洋性
        "Dm": "#009999",  # 中蓝色，冷温带湿润
        "Dw": "#99CCFF",  # 浅蓝色，大陆性季风
        "Ds": "#007CC0",  # 青蓝色，湿润大陆性
        
        # E组 - 高地和干旱大陆性气候
        "Em": "#9966CC",  # 紫色，高地草原
        "Ew": "#FFCC00",  # 金黄色，干冬草原
        "Es": "#FFCC99",  # 肉色，干夏草原
        "Ex": "#A67C29",  # 深土黄色，大陆性沙漠
        
        # F组 - 亚极地和亚寒带气候
        "Ff": "#3366CC",  # 亮蓝色，亚极地海洋性
        "Fm": "#006699",  # 深蓝紫色，北方针叶林
        "Fw": "#00CCCC",  # 深青色，极寒大陆性
        "Fs": "#000099",  # 深蓝色，亚极地-极地过渡

        # G组 - 极地气候
        "Gm": "#CC99FF",  # 浅紫色，高寒
        "Gs": "#99FFFF",  # 浅青色，极地苔原
        "Gx": "#707070"   # 灰色，冰原
    }

    # bulk operation
    @classmethod
    def classify(cls, probabilities: np.ndarray) -> list[str]:
        cls_indices = np.argmax(probabilities, axis=1)
        return [cls.class_map[i] for i in cls_indices]
    
