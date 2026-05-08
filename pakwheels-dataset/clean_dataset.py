import pandas as pd

def convert_price_to_numeric(price_str):
    """Convert prices like '33.2 lacs' or '1.2 crore' to numeric values."""
    if pd.isna(price_str) or price_str == 'Not Specified':
        return None
    
    price_str = str(price_str).strip().lower()
    price_str = price_str.replace('pkr', '').strip()
    
    if 'crore' in price_str:
        return float(price_str.replace('crore', '').strip()) * 10000000
    elif 'lac' in price_str:
        return float(price_str.replace('lac', '').replace('s', '').strip()) * 100000
    else:
        try:
            return float(price_str)
        except:
            return None

def clean_car_dataset(file_path, output_path, single_column):
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

    # 4. Clean 'Price' (Convert prices like "45 Lacs" or "1.2 Crore" to numeric)
    df['Price'] = df['Price'].apply(convert_price_to_numeric)

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

    # Keep only page_content column
    if single_column == True:
        df = df[['page_content']]

    print(f"Cleaned shape: {df.shape}")
    
    # Save the cleaned data to a new file
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

# Run the function
clean_car_dataset("pakwheels-dataset/PakWheels Dataset.csv", "pakwheels-dataset/cleaned_all_columns.csv", single_column=False)
clean_car_dataset("pakwheels-dataset/PakWheels Dataset.csv", "pakwheels-dataset/cleaned_one_column.csv", single_column=True)