from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

CORS(app)

@app.route('/predict', methods=['GET'])
def predict():
    # Obtener el ticker desde los parámetros de la URL
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({
            "status": "ERROR",
            "msg": "El parametro 'ticker' es obligatorio."
        }), 400

    try:
        # Descargar los datos
        data = yf.download(ticker, start="2022-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        if data.empty:
            raise ValueError(f"No se encontraron datos para el ticker '{ticker}'.")
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "msg": f"Error al descargar datos para el ticker '{ticker}': {str(e)}"
        }), 500

    try:
        # Ingeniería de características
        data['MA_5_dias'] = data['Close'].rolling(window=5).mean()
        data['MA_10_dias'] = data['Close'].rolling(window=10).mean()
        data['MA_20_dias'] = data['Close'].rolling(window=20).mean()
        data['Volatilidad_precio'] = data['High'] - data['Low']
        data = data.dropna()

        # Preparar datos
        caracteristicas = ['High', 'Low', 'Close', 'MA_5_dias', 'MA_10_dias', 'MA_20_dias', 'Volatilidad_precio']
        X = data[caracteristicas].values
        y = data['Close'].values

        escalador_X = MinMaxScaler()
        escalador_y = MinMaxScaler()
        X_escalado = escalador_X.fit_transform(X)
        y_escalado = escalador_y.fit_transform(y.reshape(-1, 1))
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "msg": f"Error al procesar los datos: {str(e)}"
        }), 500

    try:
        valores_reales = []
        valores_predichos = []

        for i in range(1, len(X_escalado)):
            X_train = X_escalado[:i]
            y_train = y_escalado[:i]

            modelo = LinearRegression()
            modelo.fit(X_train, y_train.ravel())

            valor_predicho_escalado = modelo.predict(X_escalado[i].reshape(1, -1))
            valor_predicho = escalador_y.inverse_transform(valor_predicho_escalado.reshape(-1, 1))[0][0]

            valores_reales.append(float(y[i]))
            valores_predichos.append(float(valor_predicho))

        ultima_caracteristica = X_escalado[-1].reshape(1, -1)
        prediccion_extra_escalada = modelo.predict(ultima_caracteristica)
        prediccion_extra = escalador_y.inverse_transform(prediccion_extra_escalada.reshape(-1, 1))[0][0]

        fecha_prediccion_extra = data.index[-1] + timedelta(days=1)
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "msg": f"Error al realizar predicciones: {str(e)}"
        }), 500

    # Preparar respuesta
    response = {
        "status": "OK",
        "msg": "Predicción realizada correctamente",
        "data": {
            "valores_reales": valores_reales,
            "valores_predichos": valores_predichos,
            "prediccion_adicional": {
                "fecha": fecha_prediccion_extra.strftime('%Y-%m-%d'),
                "valor": round(prediccion_extra, 2)
            }
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
