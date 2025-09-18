# pythonScriptsLabs
El objetivo de este repositorio es ver ejemplos de como usar la libraries de Python muy usuales en ciencia de datos.

# Como puedo usar una library?
1. Ejecutar el comando: 
    `pip install virtualenv`
2. Instalar el entorno local con el comando:
    `virtualenv local`
3. Instalar el entorno local con el comando:
    `virtualenv local`
4. Si no funciono el paso anterior, ejecutar el comando:
    `python -m virtualenv local`
5. Navegar a la carpeta creada anteriormente llamada local\Scripts\, y ejecutar el comando:
    `.\activate`
6. Si el paso anterior dio error (en entornos Windows), habilitar la posibilidad de ejecutar script desde PowerShell como admin, con el comando:
   `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
7. Volver a la carpeta donde queremos hacer las pruebas e instalar nuestra library
8. Instalar la library, como por ejemplo con el siguiente comando:
    `pip install pandas`
9. Si existe un archivo requirements.txt dentro de la carpeta donde se quiere probar una library, para poder usarla, hay que instalar las dependencias que aparecen dentro de ese archivo, con el comando:
    `pip install -r requirements.txt`