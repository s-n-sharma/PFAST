# PFAS Water Site Locator with ML Safety Prediction

A comprehensive system to locate water sites with PFAS (Per- and Polyfluoroalkyl Substances) contamination and predict water safety using Machine Learning. This system provides detailed information about PFAS levels, compound types, water site locations, and AI-powered safety assessments.

## Features

- 🔍 **Zipcode-based Search**: Find water sites near any US zipcode
- 🧠 **ML Safety Prediction**: Multi-Layer Perceptron (MLP) neural networks for safety assessment
- 📊 **PFAS Data Analysis**: Comprehensive PFAS compound information and contamination levels
- 🗺️ **Interactive Maps**: Visual representation of water sites and contamination levels
- 📈 **Advanced Visualizations**: Charts, heatmaps, and safety comparison graphs
- 🧪 **Chemical Information**: Detailed properties of detected PFAS compounds
- ⚠️ **Risk Assessment**: Automated risk categorization and priority levels
- 💡 **Smart Recommendations**: AI-generated recommendations based on safety analysis
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

Search for water sites with ML safety analysis:
```bash
python main.py --zipcode 10001
```

Search with custom radius and safety analysis:
```bash
python main.py --zipcode 90210 --radius 50
```

Search without safety analysis (faster):
```bash
python main.py --zipcode 10001 --no-safety
```

### Interactive Mode

Run the system interactively:
```bash
python main.py --interactive
```

In interactive mode, you can use these commands:
- `search 10001` - Search for sites near zipcode 10001 with safety analysis
- `search 10001 50 --no-safety` - Search with 50-mile radius without safety analysis
- `summary` - Show PFAS data summary
- `model_info` - Show ML model information
- `train_model` - Train the safety prediction model
- `help` - Show help information
- `quit` - Exit the program

### ML Model Management

Train the safety prediction model:
```bash
python main.py --train-model
```

View model information:
```bash
python main.py --model-info
```

### Data Summary

View a summary of the PFAS data:
```bash
python main.py --summary
```

## ML Safety Prediction System

### How It Works

The system uses **Multi-Layer Perceptron (MLP) neural networks** to predict water safety based on PFAS concentrations:

1. **Data Compilation**: Creates comprehensive training data with realistic PFAS concentration scenarios
2. **Feature Engineering**: Extracts relevant features from PFAS compound data
3. **Dual Model Approach**: 
   - **Classification Model**: Categorizes safety levels (Safe, Low Risk, Moderate Risk, High Risk)
   - **Regression Model**: Predicts continuous safety scores
4. **Risk Assessment**: Combines predictions with EPA guidelines for comprehensive risk analysis

### Safety Metrics

- **Safety Score**: Continuous value (lower = safer)
- **Risk Levels**: Low Risk, Moderate Risk, High Risk, Very High Risk
- **Priority Levels**: Low, Medium, High, Critical
- **Confidence Scores**: Model confidence in predictions (0-100%)

### Training Data

The system generates synthetic training data based on:
- **Real PFAS compounds** with actual chemical properties
- **EPA safety thresholds** and guidelines
- **Realistic concentration ranges** (0-200 ppb)
- **Compound interactions** and synergistic effects

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

### 3. Safety Predictor (`safety_predictor.py`)
- **MLP Neural Networks** for safety prediction
- **Dual model architecture** (classification + regression)
- **Feature engineering** and data preprocessing
- **Model training** and evaluation
- **Safety score calculation** and risk assessment

### 4. Safety Integration (`safety_integration.py`)
- Integrates ML predictions with water site data
- Generates comprehensive safety reports
- Provides risk analysis and recommendations
- Handles model training and management

### 5. Enhanced Display (`enhanced_display.py`)
- Displays ML safety analysis results
- Creates advanced visualizations
- Generates safety reports
- Shows model performance metrics

### 6. Data Display (`data_display.py`)
- Formats and presents search results
- Creates basic visualizations (charts and maps)
- Provides user-friendly output formatting

### 7. Main Application (`main.py`)
- Coordinates all system components
- Provides command-line interface
- Handles user interactions
- Manages ML model lifecycle

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

### ML Safety Predictions
- **Safety Score**: Continuous value indicating overall safety (lower = safer)
- **Risk Level**: Categorical assessment (Low/Moderate/High/Very High Risk)
- **Priority Level**: Action priority (Low/Medium/High/Critical)
- **Confidence**: Model confidence in prediction (0-100%)

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
- **ML Safety Score and Risk Level**
- **Priority Level and Recommendations**
- **Confidence Score and Risk Factors**
- Last testing date
- Specific PFAS compounds detected

## Visualizations

The system creates comprehensive visualizations:

1. **Safety Dashboard**: Multi-panel dashboard with safety metrics
2. **Risk Heatmap**: Visual representation of risk levels across sites
3. **Safety Comparison Chart**: Comparison of different safety metrics
4. **Interactive Map**: Map showing water site locations with color-coded risk levels
5. **Distance Chart**: Bar chart showing PFAS levels vs distance

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms (MLP, preprocessing)
- **geopy**: Geocoding and distance calculations
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static charts
- **requests**: HTTP requests for geocoding

### ML Model Architecture
- **Classification Model**: MLPClassifier with 3 hidden layers (100, 50, 25 neurons)
- **Regression Model**: MLPRegressor with 3 hidden layers (100, 50, 25 neurons)
- **Activation**: ReLU
- **Optimizer**: Adam with adaptive learning rate
- **Regularization**: L2 regularization (alpha=0.001)
- **Early Stopping**: Prevents overfitting

### Data Sources
- PFAS chemical properties from comprehensive database
- Sample water sites with realistic contamination scenarios
- Real-time geocoding using OpenStreetMap data
- EPA safety guidelines and thresholds

## Limitations

⚠️ **Important Notes:**
- This system uses **sample data** for demonstration purposes
- **ML predictions are for educational and research purposes**
- Always verify information with local authorities and official sources
- PFAS levels can change over time
- The system is designed for educational and research purposes

## Contributing

To extend the system:

1. **Add Real Data**: Replace sample data with real water site databases
2. **Enhance ML Models**: Add more sophisticated algorithms (Random Forest, XGBoost, etc.)
3. **Improve Visualizations**: Add more chart types and interactive features
4. **Add More PFAS Data**: Include additional compounds and properties
5. **Real-time Updates**: Integrate with real-time water quality monitoring systems

## Troubleshooting

### Common Issues

1. **"No sites found"**
   - Try increasing the search radius
   - Verify the zipcode format
   - Check if the zipcode exists

2. **ML model training errors**
   - Ensure scikit-learn is installed
   - Check for sufficient memory
   - Verify training data generation

3. **Geocoding errors**
   - Ensure internet connection
   - Try again later (rate limiting)
   - Verify zipcode format

4. **Visualization errors**
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

**Disclaimer**: This system is designed for educational and research purposes. Always consult official sources and local authorities for accurate PFAS contamination information. ML predictions should not be used as the sole basis for health or safety decisions. 