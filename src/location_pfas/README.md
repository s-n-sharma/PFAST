# PFAS Water Site Locator

A comprehensive system to locate water sites with PFAS (Per- and Polyfluoroalkyl Substances) contamination near any US zipcode. This system provides detailed information about PFAS levels, compound types, and water site locations.

## Features

- 🔍 **Zipcode-based Search**: Find water sites near any US zipcode
- 📊 **PFAS Data Analysis**: Comprehensive PFAS compound information and contamination levels
- 🗺️ **Interactive Maps**: Visual representation of water sites and contamination levels
- 📈 **Data Visualization**: Charts and graphs showing PFAS levels vs distance
- 🧪 **Chemical Information**: Detailed properties of detected PFAS compounds
- 📱 **Multiple Interfaces**: Command-line and interactive modes

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd src/location_pfas
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python main.py --help
   ```

## Usage

### Quick Start

Search for water sites near a zipcode:
```bash
python main.py --zipcode 10001
```

Search with custom radius (in miles):
```bash
python main.py --zipcode 90210 --radius 50
```

### Interactive Mode

Run the system interactively:
```bash
python main.py --interactive
```

In interactive mode, you can use these commands:
- `search 10001` - Search for sites near zipcode 10001
- `search 90210 50` - Search with 50-mile radius
- `summary` - Show PFAS data summary
- `help` - Show help information
- `quit` - Exit the program

### Data Summary

View a summary of the PFAS data:
```bash
python main.py --summary
```

## System Components

### 1. Data Processor (`pfas_data_processor.py`)
- Loads and processes PFAS chemical properties
- Creates sample water sites with realistic contamination data
- Provides data access and analysis functions

### 2. Geolocation Service (`geolocation_service.py`)
- Converts zipcodes to coordinates
- Calculates distances between locations
- Validates zipcode formats
- Caches location data for performance

### 3. Data Display (`data_display.py`)
- Formats and presents search results
- Creates visualizations (charts and maps)
- Provides user-friendly output formatting

### 4. Main Application (`main.py`)
- Coordinates all system components
- Provides command-line interface
- Handles user interactions

## Sample Data

The system includes sample water sites with realistic PFAS contamination data:

- **8 sample water sites** across different US regions
- **PFAS compounds** including PFOA, PFOS, and other common PFAS
- **Contamination levels** ranging from safe (<10 ppb) to high (>50 ppb)
- **Site types** including treatment plants, drinking water facilities, and groundwater wells

## Understanding the Results

### PFAS Levels
- **Safe**: <10 ppb (parts per billion)
- **Low**: 10-20 ppb
- **Moderate**: 20-50 ppb
- **High**: >50 ppb

### Common PFAS Compounds
- **PFOA (335-67-1)**: Perfluorooctanoic acid
- **PFOS (1763-23-1)**: Perfluorooctane sulfonic acid
- **PFBS (375-73-5)**: Perfluorobutane sulfonic acid
- **PFHxS (355-46-4)**: Perfluorohexane sulfonic acid

### Output Information
- Site name and type
- Location (city, state, zipcode)
- Distance from target zipcode
- Total PFAS level in ppb
- Last testing date
- Contamination status
- Specific PFAS compounds detected

## Visualizations

The system creates two types of visualizations:

1. **Distance Chart**: Bar chart showing PFAS levels vs distance from target location
2. **Interactive Map**: Map showing water site locations with color-coded contamination levels

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **geopy**: Geocoding and distance calculations
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static charts
- **requests**: HTTP requests for geocoding

### Data Sources
- PFAS chemical properties from comprehensive database
- Sample water sites with realistic contamination scenarios
- Real-time geocoding using OpenStreetMap data

## Limitations

⚠️ **Important Notes:**
- This system uses **sample data** for demonstration purposes
- Always verify information with local authorities and official sources
- PFAS levels can change over time
- The system is designed for educational and research purposes

## Contributing

To extend the system:

1. **Add Real Data**: Replace sample data with real water site databases
2. **Enhance Visualizations**: Add more chart types and interactive features
3. **Improve Geocoding**: Add support for international locations
4. **Add More PFAS Data**: Include additional compounds and properties

## Troubleshooting

### Common Issues

1. **"No sites found"**
   - Try increasing the search radius
   - Verify the zipcode format
   - Check if the zipcode exists

2. **Geocoding errors**
   - Ensure internet connection
   - Try again later (rate limiting)
   - Verify zipcode format

3. **Visualization errors**
   - Install all required dependencies
   - Check for display support (for maps)
   - Ensure sufficient memory for large datasets

### Getting Help

Run the help command in interactive mode:
```bash
python main.py --interactive
# Then type: help
```

## License

This project is for educational and research purposes. Please ensure compliance with local regulations when using PFAS data.

---

**Disclaimer**: This system is designed for educational and research purposes. Always consult official sources and local authorities for accurate PFAS contamination information. 