import pandas as pd
import re

def clean_car_dataset(file_path, output_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    print(f"Original shape: {df.shape}")

    # 1. Drop rows with missing critical information (Name, Price, Year)
    df.dropna(subset=['Name', 'Price', 'Year'], inplace=True)

    # 2. Clean 'Millage' (Remove "km", commas, and convert to integer)
    # Example: "45,000 km" -> 45000
    df['Millage'] = df['Millage'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df['Millage'] = pd.to_numeric(df['Millage'], errors='coerce')

    # 3. Clean 'Engine Capacity' (Remove "cc" and convert to integer)
    # Example: "1300 cc" -> 1300
    df['Engine Capacity'] = df['Engine Capacity'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df['Engine Capacity'] = pd.to_numeric(df['Engine Capacity'], errors='coerce')

    # 4. Clean 'Price' (Remove commas, "PKR", etc., and ensure it's a number)
    # Note: If your prices are text like "45 Lacs", you will need a custom function here.
    # Assuming they are numbers with commas like "4,500,000":
    df['Price'] = df['Price'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # 5. Handle missing values in text columns
    text_cols = ['Fuel', 'Transmission', 'Province', 'Color', 'Assembly', 'Body Type', 'Features']
    df[text_cols] = df[text_cols].fillna('Not Specified')

    # 6. Create the "Searchable Text" document for RAG embeddings
    # This combines the fields so the AI can read it as a single paragraph
    df['page_content'] = df.apply(lambda row: 
        f"{row['Year']} {row['Name']} ({row['Body Type']}). "
        f"Specs: {row['Engine Capacity']}cc engine, {row['Transmission']} transmission, {row['Fuel']} fuel. "
        f"Condition: {row['Millage']} km driven, Color: {row['Color']}, Assembly: {row['Assembly']}. "
        f"Location: {row['Province']}. Features include: {row['Features']}.", 
        axis=1
    )

    # Drop any rows where our conversions resulted in NaN
    df.dropna(subset=['Millage', 'Engine Capacity', 'Price'], inplace=True)

    print(f"Cleaned shape: {df.shape}")
    
    # Save the cleaned data to a new file
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

# Run the function
# clean_car_dataset("data/raw_pakwheels.csv", "data/cleaned_pakwheels.csv")