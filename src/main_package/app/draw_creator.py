import streamlit as st
import pandas as pd
import numpy as np
from src.main_package.data.load_data import load_data


def display_draw_creator():
    """
    Display a simple Streamlit page for creating the draws
    """
    male_data = st.checkbox("Create Men's Draw:")

    loaded_data = load_data(male_data)

    unique_names = np.unique(pd.concat([loaded_data["Loser"], loaded_data["Winner"]]))

    names_cache = f"overall_names_{'male' if male_data else 'female'}"
    if names_cache not in st.session_state:
        st.session_state[names_cache] = []

    col1, col2, col3 = st.columns(3)
    with col1:
        name1 = st.selectbox("Choose Player", unique_names)

        override_name1 = st.text_input("Add Unseen Player")
        if len(override_name1) > 0:
            name1 = override_name1
    with col2:
        name2 = st.selectbox("Choose Opponent", unique_names)
        override_name2 = st.text_input("Add Unseen Opponent")
        if len(override_name2) > 0:
            name2 = override_name2
    with col1:
        if st.button("Add Match"):
            if (
                name1 in st.session_state[names_cache]
                or name2 in st.session_state[names_cache]
            ):
                st.warning("Names are already chosen!!!")

            elif name1 == name2:
                st.warning("Names are the same!")
            else:
                st.session_state[names_cache].append(name1)
                st.session_state[names_cache].append(name2)

    with col3:
        names_dataframe = pd.DataFrame({"Player": st.session_state[names_cache]})
        names_dataframe["Match"] = np.cumsum(
            (np.arange(len(names_dataframe)).astype(int) + 1) % 2
        )
        st.table(names_dataframe)

    if st.button("Save"):
        saved_dataframe = names_dataframe[["Player"]]
        saved_dataframe.columns = ["player_name"]
        saved_dataframe.to_csv(
            f"src/main_package/data/draw_{'male' if male_data else 'female'}.csv",
            index=False,
        )
