from src.main_package.data.load_data import load_data


def test_load_data():
    combined_df = load_data(male_data=True)
    combined_w_df = load_data(male_data=False)

    assert len(combined_df) > 1000
    assert len(combined_w_df) > 1000
