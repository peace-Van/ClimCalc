import streamlit as st
import pandas as pd
import torch
from TorchModel import DLModel
from backend import (
    ClimateData,
    fahrenheit_to_celcius,
    celcius_to_fahrenheit,
    mm_to_inch,
    inch_to_mm,
    MONTHS,
    create_climate_chart,
    create_probability_chart,
    create_scatter_plot,
    climate_data_scraper
)
from climate_classification import DLClassification


EXAMPLE_DATA = {
    "": ["Mean Daily Minimum Temperature", "Monthly Total Precipitation", "Mean Daily Maximum Temperature"],
    "Jan": [2.7, 8.1, 9.3],
    "Feb": [4.9, 11.4, 12.1],
    "Mar": [8.4, 24.1, 16.8],
    "Apr": [12.9, 44.9, 22.5],
    "May": [17.2, 78.0, 26.3],
    "Jun": [20.5, 109.5, 28.3],
    "Jul": [22.1, 231.8, 30.0],
    "Aug": [21.7, 217.1, 29.9],
    "Sep": [18.9, 120.8, 25.7],
    "Oct": [14.7, 42.6, 20.7],
    "Nov": [9.6, 14.8, 16.0],
    "Dec": [4.2, 6.2, 10.7],
}


def get_data(data):
    return ClimateData(data)


def convert_metric(data):
    return {
        "": ["Mean Daily Minimum Temperature", "Monthly Total Precipitation", "Mean Daily Maximum Temperature"],
        **{month: [fahrenheit_to_celcius(data[month][0]), inch_to_mm(data[month][1]), fahrenheit_to_celcius(data[month][2])] for month in MONTHS}
    }


def convert_metric_to_fahrenheit(data):
    return {
        "": ["Mean Daily Minimum Temperature", "Monthly Total Precipitation", "Mean Daily Maximum Temperature"],
        **{month: [celcius_to_fahrenheit(data[month][0]), mm_to_inch(data[month][1]), celcius_to_fahrenheit(data[month][2])] for month in MONTHS}
    }


@st.cache_resource
def get_network(_weight_file: str) -> DLModel:
    model = DLModel('cpu')
    model = torch.compile(model)
    model.load_state_dict(torch.load(_weight_file, map_location=torch.device('cpu')))
    model.mode = 'inference'
    model.eval()
    return model


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="ClimCalc - Climate Classification Calculator")

    # st.markdown(
    #     """
    #     <style>
    #         .block-container {
    #             padding-top: 1rem;
    #             padding-bottom: 0rem;
    #         }
    #     </style>
    #     <script>
    #         window.onerror = function(msg, url, lineNo, columnNo, error) {
    #             // send error information to Streamlit
    #             window.parent.postMessage({
    #                 type: 'streamlit:error',
    #                 message: {
    #                     message: msg,
    #                     stack: error ? error.stack : '',
    #                     url: url,
    #                     line: lineNo,
    #                     column: columnNo        
    #                 }
    #             }, '*');
    #             return false;
    #         };
    #     </script>
    # """,
    #     unsafe_allow_html=True,
    # )

    st.title("ClimCalc - Climate Classification Calculator")

    if "unit" not in st.session_state:
        st.session_state["unit"] = True     # True for &deg;C/mm, False for &deg;F/inch

    if "data_dict" not in st.session_state:
        st.session_state["data_dict"] = EXAMPLE_DATA

    if "places" not in st.session_state:
        st.session_state["places"] = []

    model = get_network("best_model.pth")

    cols = st.columns(4)
    with cols[0]:
        with st.popover("Settings for Köppen Climate Classification", icon=":material/settings:"):
            st.radio(
                "Temperature threshold between **C** and **D**",
                ["0&deg;C", "-3&deg;C"],
                key="koppen_cd_mode",
                index=0,
                horizontal=True,
            )
            st.radio(
                "Criterion between **Bh** and **Bk**",
                [
                    "annual mean temp 18&deg;C",
                    "coldest month %s" % st.session_state["koppen_cd_mode"],
                ],
                key="koppen_kh_mode",
                index=0,
                horizontal=True,
            )

    with cols[1]:
        unit = st.radio("Unit", options=["&deg;C/mm", "&deg;F/inch"], label_visibility="collapsed", index=0, horizontal=True)
        st.session_state["unit"] = unit == "&deg;C/mm"

    with st.form("place", enter_to_submit=False):

        place_name = st.text_input("Place Name (e.g. New York City)", value="My Hometown", help="For scraping, use the place name as it appears on Wikipedia.")

        temp_dict = st.data_editor(
            st.session_state["data_dict"] if st.session_state["unit"] else convert_metric_to_fahrenheit(st.session_state["data_dict"]),
            column_config={
                "": st.column_config.TextColumn(
                    disabled=True,
                    pinned=True,
                ),
                **{month: st.column_config.NumberColumn(
                    format="%.1f",
                    required=True,
                    help="Please don't use the sorting function of the data editor.",
                ) for month in MONTHS}
            },
            hide_index=True,
        )

        cols = st.columns(4)
        with cols[0]:
            submitted_scraper = st.form_submit_button("Scrape from Wikipedia and Submit", type="primary", 
                                            help="Scrape Wikipedia for climate data using the place name. This will overwrite the manual input.")
        with cols[1]:
            submitted_manual = st.form_submit_button("Submit with manual input", type="secondary")
        with cols[2]:
            st.info("Click \"Submit with manual input\" to see how it works with the sample data.")

    if submitted_manual:

        if len(st.session_state["places"]) < 3:

            if place_name not in [place["place_name"] for place in st.session_state["places"]]:
                climate_data = get_data(temp_dict if st.session_state["unit"] else convert_metric(temp_dict))
                st.session_state["data_dict"] = temp_dict
                
                koppen = climate_data.get_koppen(0 if st.session_state["koppen_cd_mode"] == "0&deg;C" else -3, 
                                                "mean_temp" if st.session_state["koppen_kh_mode"] == "annual mean temp 18&deg;C" else "coldest_month")
                trewartha = climate_data.get_trewartha()
                thermal, aridity, prob = climate_data.get_dl(model)

                st.session_state["places"].append({
                    "place_name": place_name,
                    "climate_data": climate_data,
                    "koppen": koppen,
                    "trewartha": trewartha,
                    "thermal": thermal[0],
                    "aridity": aridity[0],
                    "probabilities": prob,
                    "dl": DLClassification.classify(prob)[0]
                })
            else:
                st.toast("Place already exists. Please enter a different place name.")

        else:
            st.toast("You can only add up to 3 places. Consider removing one using the Clear button on the chart tabs or the Clear All Locations button at the bottom.")

    if submitted_scraper:
        if len(st.session_state["places"]) < 3:
            if place_name not in [place["place_name"] for place in st.session_state["places"]]:
                with st.spinner("Scraping climate data from Wikipedia..."):
                    try:
                        climate_data = climate_data_scraper(place_name)
                        st.session_state["data_dict"] = climate_data.to_display_dict()

                        koppen = climate_data.get_koppen(0 if st.session_state["koppen_cd_mode"] == "0&deg;C" else -3, 
                                                        "mean_temp" if st.session_state["koppen_kh_mode"] == "annual mean temp 18&deg;C" else "coldest_month")
                        trewartha = climate_data.get_trewartha()
                        thermal, aridity, prob = climate_data.get_dl(model)
                        st.session_state["places"].append({
                            "place_name": place_name,
                            "climate_data": climate_data,
                            "koppen": koppen,
                            "trewartha": trewartha,
                            "thermal": thermal[0],
                            "aridity": aridity[0],
                            "probabilities": prob,
                            "dl": DLClassification.classify(prob)[0]
                        })
                        st.rerun()
                    except Exception as e:
                        st.toast(f"Error: {e}")          
                        print(e)          
            else:
                st.toast("Place already exists. Please enter a different place name.")
        else:
            st.toast("You can only add up to 3 places. Consider removing one using the Clear button on the chart tabs or the Clear All Locations button.")

    if len(st.session_state["places"]) > 0:
        thermals = [place["thermal"] for place in st.session_state["places"]]
        aridities = [place["aridity"] for place in st.session_state["places"]]
        temps = [place["climate_data"].avg_temp for place in st.session_state["places"]]
        precs = [place["climate_data"].total_prec for place in st.session_state["places"]]
        names = [place["place_name"] for place in st.session_state["places"]]
        dls = [place["dl"] for place in st.session_state["places"]]
        koppens = [place["koppen"] for place in st.session_state["places"]]
        trewarthes = [place["trewartha"] for place in st.session_state["places"]]

        df = pd.DataFrame({
            "Place Name": names,
            "Annual Mean Temperature": temps,
            "Annual Total Precipitation": precs,
            "Thermal Index": thermals,
            "Aridity Index": aridities,
            "Köppen Type": koppens,
            "Trewartha Type": trewarthes,
            "DeepEcoClimate": dls,
        })

        st.dataframe(df, hide_index=True, column_config={
            "Place Name": st.column_config.TextColumn(
                disabled=True,
                pinned=True,
            ),
            "Annual Mean Temperature": st.column_config.NumberColumn(
                format="%.2f",
            ),
            "Annual Total Precipitation": st.column_config.NumberColumn(
                format="%.2f",
            ),
            "Thermal Index": st.column_config.NumberColumn(
                format="%.2f",
            ),
            "Aridity Index": st.column_config.NumberColumn(
                format="%.2f",
            ),
            "Köppen Type": st.column_config.TextColumn(
                disabled=True,
            ),
            "Trewartha Type": st.column_config.TextColumn(
                disabled=True,
            ),
            "DeepEcoClimate": st.column_config.TextColumn(
                disabled=True,
            )})

        
        with st.container(border=True):
            cols = st.columns(3)
            with cols[0]:
                st.radio("Chart Type", options=["Climate Chart", "Class Probability Chart for DeepEcoClimate", "Thermal-Aridity Scatter Plot"], key="chart_type", label_visibility="collapsed", index=0, horizontal=True)
            with cols[1]:
                clear_all = st.button("Clear All Locations", type="primary")
                if clear_all:
                    st.session_state["places"] = []
                    st.rerun()

        places_to_remove = set()

        cols = st.columns(3, vertical_alignment="center")
        if st.session_state["chart_type"] == "Climate Chart" or st.session_state["chart_type"] == "Class Probability Chart for DeepEcoClimate":
            for place_idx in range(3):
                if place_idx < len(st.session_state["places"]):
                    with cols[place_idx]:
                        with st.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.checkbox(
                                    "Auto scale axes",
                                    value=False,
                                    key=f"auto_scale_{place_idx}_climate",
                                    disabled=st.session_state["chart_type"] == "Class Probability Chart for DeepEcoClimate",
                                )
                            with col2:
                                st.checkbox(
                                    "July first",
                                    value=False,
                                    key=f"july_first_{place_idx}_climate",
                                    disabled=st.session_state["chart_type"] == "Class Probability Chart for DeepEcoClimate",
                                )

                            if st.session_state["chart_type"] == "Class Probability Chart for DeepEcoClimate":
                                thermal = st.session_state["places"][place_idx]["thermal"]
                                aridity = st.session_state["places"][place_idx]["aridity"]
                                probs = st.session_state["places"][place_idx]["probabilities"]
                                fig = create_probability_chart(
                                    probabilities=probs,
                                    class_map=DLClassification.class_map,
                                    color_map=DLClassification.color_map,
                                    title=st.session_state["places"][place_idx]["place_name"],
                                    subtitle=f"Thermal Index: {thermal:.2f}, Aridity Index: {aridity:.2f}, DeepEcoClimate: {st.session_state['places'][place_idx]['dl']}",
                                )
                            else:
                                fig = create_climate_chart(
                                    st.session_state["places"][place_idx]["climate_data"],
                                    title=st.session_state["places"][place_idx]["place_name"],
                                    subtitle=f"Köppen: {st.session_state['places'][place_idx]['koppen']}, Trewartha: {st.session_state['places'][place_idx]['trewartha']}, DeepEcoClimate: {st.session_state['places'][place_idx]['dl']}",
                                    july_first=st.session_state[f"july_first_{place_idx}_climate"],
                                    unit=not st.session_state["unit"],
                                    auto_scale=st.session_state[f"auto_scale_{place_idx}_climate"],
                                )
                                
                            st.plotly_chart(fig, use_container_width=True)

                            with col3:
                                delete = st.button(
                                    "Clear",
                                    key=f"clear_{place_idx}_climate",
                                    type="primary",
                                )
                                if delete:
                                    places_to_remove.add(place_idx)

                else:
                    st.empty()
        
        elif st.session_state["chart_type"] == "Thermal-Aridity Scatter Plot":
            with cols[1]:
                fig = create_scatter_plot(
                    places=st.session_state["places"],
                    class_centers=model.cluster.centers.numpy(),
                )
                st.plotly_chart(fig, use_container_width=True)
            with cols[2]:
                st.info("Use full screen mode of the chart for better visibility. \n\n DeepEcoClimate uses 60 features to classify climate types, \
                        and the scatter plot only shows the first 2 principal components. So sometimes you may find a point appears \
                        far away from the cluster center. They can be close in other 58 dimensions.")
        
        if places_to_remove:
            st.session_state["places"] = [st.session_state["places"][i] for i in range(len(st.session_state["places"])) if i not in places_to_remove]
            st.rerun()

        # if len(st.session_state["places"]) > 0:
        #     clear_all = st.button("Clear All", type="primary")
        #     if clear_all:
        #         st.session_state["places"] = []
        #         st.rerun()

        st.divider()
        st.page_link("https://climviz.streamlit.app/", label="ClimViz - Global Map over years", icon=":material/globe:")
