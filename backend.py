import numpy as np
from climate_classification import KoppenClassification, TrewarthaClassification, DLClassification
from TorchModel import DLModel
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from bs4 import BeautifulSoup
import re

import sys
sys.path.insert(0, './custom_wikipedia')
import wikipedia


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

TEMP_RANGE = [
    [-25, 40],  # °C
    [-8, 96],   # °F
]  
TEMP_TICKVALS = [
    [-20, -10, 0, 10, 20, 30, 40],  # °C
    [0, 16, 32, 48, 64, 80, 96],   # °F
]  
PREC_RANGE = [
    [0, 390],  # mm
    [0, 13],   # inch
]  
PREC_TICKVALS = [
    [30, 90, 150, 210, 270, 330, 390],  # mm
    [1, 3, 5, 7, 9, 11, 13],            # inch
]  


def fahrenheit_to_celcius(temp):
    return (temp - 32) * (5 / 9)


def celcius_to_fahrenheit(temp):
    return temp * (9 / 5) + 32


def mm_to_inch(prec):
    return prec / 25.4


def inch_to_mm(prec):
    return prec * 25.4


class ClimateData:
    __slots__ = ["data"]

    data: np.ndarray

    def __init__(self, data_dict: dict[str, any]):
        self.data = np.zeros((3, 12), dtype=np.float32)
        if len(data_dict) == 13:
            for month in MONTHS:
                self.data[0, MONTHS.index(month)] = data_dict[month][0]
                self.data[1, MONTHS.index(month)] = data_dict[month][1]
                self.data[2, MONTHS.index(month)] = data_dict[month][2]
        elif len(data_dict) == 3:
            self.data[0, :] = data_dict["tmn"][:-1]
            self.data[1, :] = data_dict["pre"][:-1]
            self.data[2, :] = data_dict["tmx"][:-1]

    def __str__(self):
        return f"ClimateData(tmn={self.data[0, :]}, pre={self.data[1, :]}, tmx={self.data[2, :]})"
    
    def to_display_dict(self):
        return {
            "": ["Mean Daily Minimum Temperature", "Monthly Total Precipitation", "Mean Daily Maximum Temperature"],
            **{month: [self.data[0, MONTHS.index(month)], self.data[1, MONTHS.index(month)], self.data[2, MONTHS.index(month)]] for month in MONTHS}
        }
    
    @property
    def avg_temp(self):
        return (self.data[0, :] + self.data[2, :]).sum() / 24
    
    @property
    def total_prec(self):
        return self.data[1, :].sum()

    def get_dl_data(self) -> np.ndarray:
        return self.data[np.newaxis, :, :]
    
    def get_classic_data(self) -> np.ndarray:
        return np.vstack([(self.data[0, :] + self.data[2, :]) / 2, self.data[1, :]])
    
    def get_koppen(self, cd_threshold: float, kh_mode: str) -> str:
        return KoppenClassification.classify(self.get_classic_data(), cd_threshold, kh_mode)
    
    def get_trewartha(self) -> str:
        return TrewarthaClassification.classify(self.get_classic_data())
    
    def get_dl(self, model: DLModel) -> tuple[float, float, np.ndarray]:
        return model(self.get_dl_data())
    

def climate_data_scraper(place_name: str) -> ClimateData:
    try:
        page = wikipedia.page(place_name)
    except wikipedia.exceptions.DisambiguationError as e:
        place_name = e.options[0]
        page = wikipedia.page(place_name)
    except wikipedia.exceptions.PageError:
        raise Exception(f"{place_name} not found")
    except Exception as e:
        raise Exception(f"Network error: {e}")
    
    soup = BeautifulSoup(page.html(), "lxml")
    tables = soup.find_all("table", {"class": "wikitable"})
    if not tables:
        raise Exception(f"No climate data table found for {place_name}")
    
    def parse_value(value: str, label: str) -> float:
        value = value.strip()
        value = value.replace('−', '-')
        value = value.replace(',', '')

        pattern = r'([+-]?\d+(?:\.\d+)?)\(([+-]?\d+(?:\.\d+)?)\)'

        match = re.match(pattern, value)
        if match:
            num1 = float(match.group(1))
            num2 = float(match.group(2))
        else:
            raise ValueError(f"cannot parse value: {value}")

        if "°f (°c)" in label or "inches (mm)" in label:
            return num2
        elif "°c (°f)" in label or "mm (inches)" in label:
            return num1

    for table in tables:
        rows = table.find_all("tr")
        row_data = {}
        for row in rows:
            th = row.find("th")
            if not th:
                continue
            label = th.get_text(separator=' ', strip=True).lower()
            if "mean daily minimum" in label:
                row_data["tmn"] = [parse_value(td.get_text(strip=True), label) for td in row.find_all("td")]
            elif "mean daily maximum" in label:
                row_data["tmx"] = [parse_value(td.get_text(strip=True), label) for td in row.find_all("td")]
            elif "average precipitation mm" in label or "average precipitation inches" in label or "average rainfall mm" in label or "average rainfall inches" in label:
                row_data["pre"] = [parse_value(td.get_text(strip=True), label) for td in row.find_all("td")]
        if len(row_data) == 3:
            return ClimateData(row_data)

    raise Exception(f"Climate data for {place_name} is not complete on Wikipedia")
    

def create_climate_chart(
    climate_data: ClimateData,
    title: str,
    subtitle: str,
    july_first: bool,
    unit: bool,
    auto_scale: bool,
) -> go.Figure:
    """
    create climate chart

    Args:
        climate_data: ClimateData object
        location: coordinates (lat, lon)
        local_lang: whether to use local language
        july_first: whether to start from July
        unit: whether to use °F/inch
        auto_scale: whether to auto scale
        locationService: LocationService object
    Returns:
        plotly.graph_objects.Figure object
    """
    # prepare data
    prec = climate_data.data[1, :]  # monthly precipitation
    # evap = climate_data.pet  # monthly evaporation
    tmax = climate_data.data[2, :]  # daily maximum temperature
    tmin = climate_data.data[0, :]  # daily minimum temperature
    temp = (tmax + tmin) / 2  # monthly mean temperature

    months = MONTHS
    # if start from July, adjust the data order
    if july_first:
        months = MONTHS[6:] + MONTHS[:6]
        temp = np.concatenate((temp[6:], temp[:6]))
        prec = np.concatenate((prec[6:], prec[:6]))
        tmax = np.concatenate((tmax[6:], tmax[:6]))
        tmin = np.concatenate((tmin[6:], tmin[:6]))

    if unit:
        temp = celcius_to_fahrenheit(temp)
        tmax = celcius_to_fahrenheit(tmax)
        tmin = celcius_to_fahrenheit(tmin)
        prec = mm_to_inch(prec)

    fig = go.Figure()

    # add precipitation bar chart
    fig.add_trace(
        go.Bar(
            x=months,
            y=prec,
            name="Precipitation",
            marker_color="rgba(0, 135, 189, 0.5)",
            yaxis="y2",
            showlegend=False,
            hovertemplate="(%{x}, %{y:.1f})",
        )
    )

    # # add evaporation bar chart
    # fig.add_trace(
    #     go.Bar(
    #         x=months,
    #         y=evap,
    #         name="Evaporation",
    #         marker_color="rgba(255, 211, 0, 0.5)",
    #         yaxis="y2",
    #         showlegend=False,
    #         hovertemplate="(%{x}, %{y:.1f})",
    #     )
    # )

    # add temperature line chart
    fig.add_trace(
        go.Scatter(
            x=months,
            y=temp,
            name="Mean Temperature",
            line=dict(color="rgba(196, 2, 52, 0.8)", width=2),
            mode="lines+markers",
            showlegend=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=tmax - temp,
                arrayminus=temp - tmin,
            ),
            hovertemplate="(%{x}, %{y:.1f})",
        )
    )

    # add temperature range
    fig.add_trace(
        go.Scatter(
            x=months,
            y=tmax,
            mode="markers",
            name="Daily Maximum",
            marker=dict(color="rgba(196, 2, 52, 0.8)", size=8),
            showlegend=False,
            hovertemplate="(%{x}, %{y:.1f})",
        )
    )

    # add temperature range
    fig.add_trace(
        go.Scatter(
            x=months,
            y=tmin,
            mode="markers",
            name="Daily Minimum",
            marker=dict(color="rgba(196, 2, 52, 0.8)", size=8),
            showlegend=False,
            hovertemplate="(%{x}, %{y:.1f})",
        )
    )

    # update layout
    fig.update_layout(
        title=dict(
            text=title,
            subtitle=dict(text=subtitle, font=dict(size=13)),
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=14),
        ),
        margin=dict(t=60, l=60, r=60, b=30),  # add top margin to leave space for the title
        height=400,  # set the chart height, ensure the chart size is fixed
        yaxis=dict(
            title="Temperature (°C)" if not unit else "Temperature (°F)",
            range=TEMP_RANGE[unit],
            autorange=auto_scale,
            showgrid=not auto_scale,
            gridcolor="lightgray" if not auto_scale else None,
            zeroline=(not unit),
            zerolinecolor="black",
            linecolor="rgb(196, 2, 52)",
            tickcolor="rgb(196, 2, 52)",
            tickfont=dict(color="rgb(196, 2, 52)"),
            tickvals=TEMP_TICKVALS[unit] if not auto_scale else None,
        ),
        yaxis2=dict(
            title=(
                "Precipitation/Evaporation (mm)"
                if not unit
                else "Precipitation/Evaporation (inch)"
            ),
            range=PREC_RANGE[unit],
            autorange=auto_scale,
            showgrid=not auto_scale,
            gridcolor="lightgray" if not auto_scale else None,
            overlaying="y",
            side="right",
            zeroline=True,
            zerolinecolor="black",
            linecolor="rgb(0, 135, 189)",
            tickvals=PREC_TICKVALS[unit] if not auto_scale else None,
            tickcolor="rgb(0, 135, 189)",
            tickfont=dict(color="rgb(0, 135, 189)"),
        ),
        xaxis=dict(gridcolor="lightgray"),
        shapes=(
            [
                dict(
                    type="line",
                    x0=0,
                    x1=1,
                    y0=32 if unit else 0,
                    y1=32 if unit else 0,
                    xref="paper",
                    yref="y",
                    line=dict(color="black", width=0.5),
                )
            ]
            if unit
            else None
        ),  # add this line only when unit is Fahrenheit
        barmode="overlay",
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig


def create_probability_chart(
    probabilities: np.ndarray,
    class_map: list[str],
    color_map: dict[str, str],
    title: str,
    subtitle: str,
) -> go.Figure:
    """
    create climate type probability distribution chart

    Args:
        probabilities: probability array, shape is (n_classes,)
        class_map: climate type name list
        color_map: climate type color mapping dictionary
        title: title
        subtitle: sub-title
    Returns:
        plotly.graph_objects.Figure object
    """
    # get the indices of the top 3 types
    probabilities = probabilities.squeeze()
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    
    # prepare data
    classes = [class_map[i] for i in top_3_indices]
    probs = probabilities[top_3_indices]
    colors = [color_map[cls] for cls in classes]

    fig = go.Figure()

    # add probability bar chart
    fig.add_trace(
        go.Bar(
            x=classes,
            y=probs,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],  # show percentage
            textposition="auto",
            textfont=dict(size=13),
            hoverinfo="skip",  # disable hover tooltip
        )
    )

    # update layout
    fig.update_layout(
        title=dict(
            text=title,
            subtitle=dict(text=subtitle, font=dict(size=13)),
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=14),
        ),
        margin=dict(t=60, l=60, r=60, b=30),
        height=400,
        yaxis=dict(
            title="Probability",
            range=[0, 1],
            tickformat=".0%",
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="black",
            tickfont=dict(size=13),
        ),
        xaxis=dict(
            tickfont=dict(size=13),
        ),
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig


def create_scatter_plot(
    places: list[dict[str, any]],
    class_centers: np.ndarray,
) -> go.Figure:
    thermals = [place["thermal"] for place in places]
    thermals.extend(class_centers[:, 0])
    aridities = [place["aridity"] for place in places]
    aridities.extend(class_centers[:, 1])
    names = [place["place_name"] + " (" + place["dl"] + ")" for place in places]
    names.extend(DLClassification.class_map)
    colors = [DLClassification.color_map[place["dl"]] for place in places]
    colors.extend([DLClassification.color_map[cls] for cls in DLClassification.class_map])
    
    df = pd.DataFrame({
        "Thermal Index": thermals,
        "Aridity Index": aridities,
        "Place": names,
    })
    
    fig = px.scatter(
        df,
        x="Thermal Index",
        y="Aridity Index",
        text="Place"
    )
    fig.update_traces(
        textposition="middle right",
        marker_color=colors,
        marker=dict(size=8, opacity=0.8),
        textfont=dict(size=12, color='black'),
        hovertemplate="%{text}<br>Thermal Index: %{x:.2f}<br>Aridity Index: %{y:.2f}<extra></extra>"
    )

    fig.add_shape(type="line", x0=-0.4, x1=-0.4, y0=-0.8, y1=0.8, line=dict(color="black", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0.4, x1=0.4, y0=-0.8, y1=0.8, line=dict(color="black", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-0.8, x1=0.8, y0=-0.4, y1=-0.4, line=dict(color="black", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-0.8, x1=0.8, y0=0.4, y1=0.4, line=dict(color="black", width=1, dash="dash"))

    fig.update_layout(
        xaxis=dict(
            range=[-0.8, 0.8],
            zeroline=False,
            showgrid=False,
            title="Thermal Index",
            title_font=dict(size=16, color="black"),
            tickfont=dict(size=12, color="black"),
        ),
        yaxis=dict(
            range=[-0.8, 0.8],
            zeroline=False,
            showgrid=False,
            title="Aridity Index",
            title_font=dict(size=16, color="black"),
            tickfont=dict(size=12, color="black"),
        ),
        plot_bgcolor="white",
        margin=dict(t=0, l=0, r=0, b=0),
    )
    
    return fig


# if __name__ == "__main__":
#     data = climate_data_scraper("New York")
#     print(data)
