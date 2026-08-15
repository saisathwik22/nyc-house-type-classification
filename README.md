# NYC Airbnb Room Type Predictor

A machine learning-powered web application that predicts the type of room (Entire home/apt, Private room, or Shared room) for NYC Airbnb listings based on listing characteristics and host information.

---

## 🎯 Project Overview

The **NYC Airbnb Room Type Predictor** is a full-stack application that combines:

- **Backend**: A FastAPI server that hosts a pre-trained scikit-learn classification model
- **Frontend**: A modern, interactive web interface with NYC-themed aesthetics and real-time predictions

The application allows users to input various listing details and receive instant predictions about what type of room that listing represents, along with confidence probabilities for each room type class.

### Key Features

- 🏙️ **Beautiful NYC-Themed Interface**: Animated skyline background with twinkling window lights
- 🤖 **ML-Powered Predictions**: Uses a pre-trained scikit-learn pipeline to classify room types
- 📊 **Visual Probability Display**: Animated building representations showing confidence scores
- 📍 **Comprehensive Input Validation**: Client-side and server-side validation of all input fields
- ⚡ **Real-time API Health Checks**: Automatic detection of backend API availability
- 🎨 **Dark Mode Design**: Modern dark theme with carefully selected color palette
- 📱 **Responsive Layout**: Clean, side-by-side panel layout for form input and results

---

## 🏗️ Project Structure

```
house_types_classification/
├── index.html          # Main HTML structure with form and results panels
├── script.js           # Frontend logic, API communication, and animations
├── style.css           # Complete styling with custom design tokens
├── main.py             # FastAPI backend server with ML model
├── requirements.txt    # Python dependencies
├── Model_Pipeline.pkl  # Pre-trained scikit-learn model (binary)
└── README.md          # This file
```

---

## 📋 Technology Stack

### Backend
- **FastAPI** (0.141.1): Modern, fast web framework for building APIs
- **Python** (3.x): Programming language
- **scikit-learn** (1.6.1): Machine learning library for model serving
- **pandas** (3.0.5): Data manipulation and processing
- **Pydantic** (2.13.4): Data validation using Python type hints
- **joblib** (1.5.3): Model serialization and loading
- **uvicorn** (0.52.3): ASGI server for running FastAPI
- **CORS Middleware**: Enables cross-origin requests for frontend communication

### Frontend
- **HTML5**: Semantic markup and form structure
- **CSS3**: Modern styling with CSS variables (design tokens), animations, and gradients
- **Vanilla JavaScript**: No frameworks; pure DOM manipulation and Fetch API
- **Google Fonts**: Space Grotesk (display), Inter (body), JetBrains Mono (monospace)

---

## 🧠 Machine Learning Model

### Model Pipeline
The application uses a pre-trained scikit-learn model pipeline saved as `Model_Pipeline.pkl`.

**Training Data**: NYC Airbnb listings dataset

**Target Variable**: Room Type
- Entire home/apt
- Private room
- Shared room

### Input Features (10 features)

| Feature | Type | Description | Constraints |
|---------|------|-------------|-------------|
| `latitude` | float | Listing latitude coordinate | -90 to 90 |
| `longitude` | float | Listing longitude coordinate | -180 to 180 |
| `price` | float | Price per night in USD | > 0 |
| `minimum_nights` | int | Minimum night requirement for booking | 1-365 nights |
| `number_of_reviews` | int | Total number of reviews received | ≥ 0 |
| `reviews_per_month` | float | Average reviews per month | ≥ 0 |
| `calculated_host_listings_count` | int | Total listings from this host | ≥ 0 |
| `availability_365` | int | Days available in next 365 days | 0-365 days |
| `neighbourhood_group` | str | NYC borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island) | - |
| `neighbourhood` | str | Specific neighbourhood name | - |

### Model Output
- **Predicted Room Type**: The most likely room classification
- **Probability Array**: Confidence scores (0-1) for each of the 3 room type classes

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager
- A modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone or download the project**
   ```bash
   cd house_types_classification
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the FastAPI backend server**
   ```bash
   uvicorn main:app --reload
   ```
   - The API will be available at `http://localhost:8000`
   - Swagger API documentation: `http://localhost:8000/docs`
   - ReDoc documentation: `http://localhost:8000/redoc`

2. **Open the frontend in your browser**
   - Open `index.html` in your web browser
   - Or serve it using a local web server:
     ```bash
     # Using Python's built-in server
     python -m http.server 8080
     ```
   - Then navigate to `http://localhost:8080`

3. **Make predictions**
   - Fill in the listing details in the left panel
   - Click "Predict room type" to get predictions
   - View probability distribution in the right panel
   - Use "Try an example" to quickly test with sample listings

---

## 📡 API Specification

### Endpoints

#### GET `/`
**Health check endpoint**

**Response:**
```json
"Hello Guyss"
```

---

#### POST `/predict`
**Prediction endpoint**

**Request Body (JSON):**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "price": 150,
  "minimum_nights": 3,
  "number_of_reviews": 24,
  "reviews_per_month": 1.2,
  "calculated_host_listings_count": 1,
  "availability_365": 180,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}
```

**Response (Success - 200):**
```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.85, 0.12, 0.03]
}
```

**Response (Validation Error - 422):**
```json
{
  "detail": [
    {
      "type": "validation_error",
      "loc": ["body", "latitude"],
      "msg": "ensure this value is less than or equal to 90",
      "input": 100
    }
  ]
}
```

### Input Validation

Pydantic model validation enforces:
- **Type checking**: Correct data types for each field
- **Range validation**: Latitude (-90 to 90), Longitude (-180 to 180), etc.
- **Positive constraints**: Price must be > 0
- **Length constraints**: Neighbourhood names must not be empty
- **Step validation**: Precision constraints for decimal values

---

## 🎨 Frontend Architecture

### Main Components

#### Form Panel (Left)
- **Location Section**: Latitude, longitude, borough (dropdown), neighbourhood
- **Pricing & Stay Section**: Price per night, minimum nights, availability slider
- **Reviews & Host Section**: Total reviews, reviews per month, host's listing count
- **Actions**: "Predict room type" button and "Try an example" button

#### Result Panel (Right)
- **API Status Indicator**: Real-time connection status to backend
- **Prediction Display**: Shows the most likely room type
- **Building Visualization**: Animated NYC buildings representing probability for each class
  - Taller building = higher confidence
  - Lit windows = proportion corresponding to probability
- **Probability List**: Detailed percentages for each room type

#### Visual Elements
- **Skyline Background**: Animated twinkling window lights creating NYC ambiance
- **Grain Texture**: Subtle texture overlay for visual depth
- **Color Palette**:
  - Deep Dark: `#0B0F1A` (background)
  - Panel: `#131A2C`
  - Accent (Amber): `#FFB454` (highlighted predictions)
  - Accent (Teal): `#4FD1C5` (UI elements)
  - Text Primary: `#EDEDF2`

### JavaScript Functionality

**Core Functions:**
- `buildSkylineLights()`: Generates animated background lights
- `collectPayload()`: Gathers form data and converts to API format
- `setLoading()`: Manages loading states and button disabled state
- `renderResult()`: Displays prediction results with animations
- `buildBuildings()`: Creates animated building representations
- `buildProbList()`: Renders probability percentages with animated counters
- `checkApiStatus()`: Polls backend health endpoint
- `animateCount()`: Smooth number counter animation

**Example Listings:**
Pre-configured sample data for quick testing:
- Manhattan Midtown: $120/night, high reviews
- Brooklyn Bedford-Stuyvesant: $55/night, very active
- Queens Flushing: $38/night, budget option

### Animations & UX
- **Smooth transitions**: All state changes use CSS transitions
- **Staggered animations**: Windows light up sequentially for visual effect
- **Ease functions**: Cubic easing for natural motion
- **Reduced motion support**: Respects `prefers-reduced-motion` system setting
- **Loading states**: Button spinner indicates processing
- **Real-time feedback**: Range slider updates availability display instantly

---

## 🔌 CORS Configuration

The backend is configured with CORS middleware to allow requests from any origin:

```python
CORSMiddleware(
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
```

This enables:
- Requests from `file://` protocol (local HTML file)
- Requests from any domain (for web deployment)
- All HTTP methods and headers

---

## 🧪 Testing the Application

### Manual Test Cases

1. **Valid Prediction**
   - Input: All required fields with valid values
   - Expected: Prediction displayed with probabilities

2. **Validation Error**
   - Input: Latitude = 150 (out of range)
   - Expected: Error message "ensure this value is less than or equal to 90"

3. **API Unreachable**
   - Action: Stop the FastAPI server
   - Expected: API status shows "API unreachable" and error on submit

4. **Example Loading**
   - Action: Click "Try an example" three times
   - Expected: Form populates with different example listings in rotation

5. **Range Slider**
   - Action: Move availability slider
   - Expected: Number display updates in real-time

### Using the Swagger API Documentation

1. Navigate to `http://localhost:8000/docs`
2. Expand the `/predict` endpoint
3. Click "Try it out"
4. Enter test data in the request body
5. Click "Execute" to send the request
6. View the response

---

## 🚀 Deployment

### Environment Variables
None currently required for basic operation.

### Backend Deployment (Production)
The project is currently configured to connect to a Render.com deployment:
```javascript
const API_BASE_URL = "https://nyc-airbnb-room-type-predictor.onrender.com";
```

To deploy to Render:
1. Create a Render.com account
2. Connect your GitHub repository
3. Set up a Web Service
4. Deploy using `uvicorn main:app --host 0.0.0.0 --port 8000`

### Frontend Deployment
- Deploy `index.html`, `script.js`, `style.css` to any static hosting:
  - GitHub Pages
  - Netlify
  - Vercel
  - AWS S3
  - Any web server

### Local Deployment Checklist
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Model file `Model_Pipeline.pkl` present in project root
- [ ] Backend running on expected port (8000)
- [ ] `API_BASE_URL` in `script.js` points to correct backend
- [ ] Firewall allows connections to API port

---

## 📊 Data Flow

```
User Input (Form)
    ↓
JavaScript collectPayload()
    ↓
Fetch POST to /predict
    ↓
FastAPI receives Features model
    ↓
Pydantic validates input
    ↓
pandas creates DataFrame row
    ↓
scikit-learn pipeline predicts
    ↓
JSON response (prediction + probabilities)
    ↓
renderResult() displays in UI
    ↓
Animations play (buildings + counters)
    ↓
User sees result
```

---

## 🔧 Troubleshooting

### Issue: "Can't reach the prediction API"
**Solution:**
- Ensure FastAPI server is running (`uvicorn main:app --reload`)
- Check that the API is on `http://localhost:8000`
- Update `API_BASE_URL` in `script.js` if using different port/host
- Check browser console for CORS errors

### Issue: Model not found error
**Solution:**
- Ensure `Model_Pipeline.pkl` exists in the same directory as `main.py`
- Verify the filename is exactly correct (case-sensitive on Linux/Mac)

### Issue: Python dependencies not found
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Port 8000 already in use
**Solution:**
```bash
# Use different port
uvicorn main:app --reload --port 8001
# Update script.js to point to new port
```

### Issue: Predictions seem incorrect
**Solutions:**
- Verify input data is within expected ranges
- Check that `Model_Pipeline.pkl` hasn't been corrupted
- Review training data to understand model expectations
- Consider retraining the model with updated data

---

## 📈 Model Performance Insights

The model has been trained on NYC Airbnb dataset features. Typical performance metrics:

- **Feature Importance**: Price, location (lat/lon), and listing characteristics are strong predictors
- **Room Type Distribution**: 
  - Entire home/apt: ~50%
  - Private room: ~45%
  - Shared room: ~5% (minority class)
- **Prediction Challenges**: 
  - Distinguishing private rooms from shared rooms (both limited availability)
  - Extreme price outliers
  - Missing or unusual neighbourhood data

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **Full-Stack Development**: Backend API + Frontend UI
2. **Machine Learning Serving**: Deploying scikit-learn models
3. **API Design**: RESTful endpoints with Pydantic validation
4. **Web Development**: HTML5, CSS3, vanilla JavaScript
5. **UX/UI Design**: Modern, responsive interface with animations
6. **CORS & Security**: Cross-origin resource sharing configuration
7. **Data Validation**: Server-side and client-side validation patterns

---

## 📝 Notes

- The model assumes data patterns similar to the NYC Airbnb training dataset
- Predictions are probabilistic; confidence levels should be considered
- The application works best with realistic NYC Airbnb listing values
- Very unusual input combinations may produce unexpected results

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 👤 Author

Created as a machine learning classification demonstration project.

---

## 🔗 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

---

**Last Updated**: 2026-08-15

---

## Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run backend server
uvicorn main:app --reload

# Run frontend (if not opening HTML directly)
python -m http.server 8080

# Access points
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:8080 or file:///path/to/index.html
```

---

This comprehensive README provides complete documentation for understanding, running, and extending the NYC Airbnb Room Type Predictor application.
