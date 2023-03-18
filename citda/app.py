
from shap_calculator import ShapCalculator
import os

def get_menu():
    menu = """Please select one of the following options:
            1) Calculate/Load SHAP values.
            2) Create LIWC/SHAP scores JSON file.
            3) Calculate CITDA scores.
            4) Exit.

            Your selection: """
    return menu
    
if __name__ == "__main__":
    menu = get_menu()
    shap_values_filename = 'shap_values.pkl'
    shap_calculator = ShapCalculator(
                data_file="https://github.com/noghte/datasets/raw/main/liwc.csv",
                text_column="text", 
                label_column="emotion",
                shap_values_filename = "shap_values_filename.pickle",
                model_name="nateraw/bert-base-uncased-emotion",
                tokenizer_name="nateraw/bert-base-uncased-emotion",
                Features= ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])
    
    while (user_input := input(menu)) != "4":
        if user_input == "1": # Calculate/Load SHAP values
            if os.path.isfile(shap_values_filename): 
                shap_values = shap_calculator.load_shap_values()
                print(f"Loading SHAP values from {shap_values_filename}.")
            else:
                shap_calculator.calculate()
        elif user_input == "2": # Create LIWC/SHAP scores JSON file
            shap_calculator.create_liwc_shap(filename="liwc_shap_scores.json")
        elif user_input == "3": # Calculate CITDA scores
            print("Calculating CITDA scores...")
            