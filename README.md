# Shoe Brand Detection

## Descripción
Este proyecto utiliza una red neuronal convolucional (CNN) para identificar la marca de zapatos a partir de imágenes. Incluye una interfaz web para permitir a los usuarios subir imágenes y obtener predicciones.

## Estructura del Proyecto
- `app.py`: Backend Flask para manejar la carga de imágenes y realizar predicciones.
- `model/shoe_brand_model.h5`: Modelo entrenado.
- `static/`: Archivos estáticos como CSS.
- `templates/`: Plantillas HTML.
- `uploads/`: Carpeta para almacenar imágenes subidas.
- `data/`: Datos de entrenamiento, validación y prueba.
- `README.md`: Información del proyecto.

## Configuración
1. **Instalar dependencias**:
   ```bash
   pip install tensorflow flask
