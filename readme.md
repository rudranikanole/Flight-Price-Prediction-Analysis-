✈️ Flight Price Prediction & Analysis
This project predicts flight ticket prices based on factors like duration, days left, departure hour, and arrival hour using Machine Learning models. The web app is built using Flask and provides a simple UI to get flight price predictions.

📌 Dataset Used
The dataset used for training is a cleaned version of flight price data, which includes:

duration: Flight duration in hours
days_left: Days left before departure
departure_hour: Departure time in 24-hour format
arrival_hour: Arrival time in 24-hour format
price: Actual flight price (Target Variable)
📂 Dataset File: data/Processed_Dataset.csv

🔧 Technologies Used (Tech Stack)
✅ Frontend: HTML, CSS
✅ Backend: Flask (Python-based web framework)
✅ Machine Learning: Scikit-learn, LightGBM
✅ Data Processing: Pandas, NumPy
✅ Visualization: Matplotlib, Seaborn
✅ Model Training & Selection: GridSearchCV, RandomizedSearchCV

⚡ How to Run the Project?
1️⃣ Clone the Repository:

bash
Copy
Edit
git clone https://github.com/yourusername/flight-price-prediction.git
cd flight-price-prediction
2️⃣ Create and Activate a Virtual Environment (Optional but Recommended)

bash
Copy
Edit
python -m venv flight_env  
source flight_env/bin/activate  # For macOS/Linux  
flight_env\Scripts\activate  # For Windows  
3️⃣ Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the Flask Application:

bash
Copy
Edit
python app.py
5️⃣ Open in Browser:
Go to:

cpp
Copy
Edit
http://127.0.0.1:5000/
📂 Project Structure
bash
Copy
Edit
/flight_price_prediction/
│── /static/                # CSS & Images  
│   ├── style.css  
│   ├── airplane.png  (Icon)  
│   ├── background.jpg  (Background Image)  
│── /templates/             # HTML Templates  
│   ├── index.html  
│── /data/                  # Dataset Folder  
│   ├── Processed_Dataset.csv  
│── /models/                # Trained Models (Optional)  
│── app.py                  # Flask Backend  
│── train_model.py          # Model Training Script  
│── eda.py                  # Data Analysis & Visualization  
│── my_script.py            # Data Processing Script  
│── requirements.txt        # Python Dependencies  
│── README.md               # Project Documentation  
🚀 Future Enhancements
🔹 Improve model accuracy with advanced ML techniques.
🔹 Add real-time flight data via API integration.
🔹 Deploy the app on AWS/GCP/Heroku for wider accessibility.

