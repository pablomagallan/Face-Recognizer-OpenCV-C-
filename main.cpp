#include <opencv2\opencv.hpp>
#include <ctime>

using namespace cv;
using namespace std;

CascadeClassifier faceDetector;
CascadeClassifier lEyeDetector;
CascadeClassifier rEyeDetector;

VideoCapture capture;

int REDUCED_SIZE = 480;

//Obtiene las coordenas de los ojos y el rostro, devuelve true si las encuentra
bool EncontrarRostroYOjos(Mat& frame, Rect& rostro, Rect& lEye, Rect& rEye)
{
	vector<Rect> faces;

    faceDetector.detectMultiScale(frame, faces, 1.1, 3,
                                  CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH);

	float EYE_SX = 0.12f;
    float EYE_SY = 0.17f;
    float EYE_SW = 0.37f;
    float EYE_SH = 0.36f;

	if(faces.size() == 1)
	{
		rostro = faces[0];
		Mat face = frame(rostro);

		int leftX = cvRound(face.cols * EYE_SX);
		int topY = cvRound(face.rows * EYE_SY);
		int widthX = cvRound(face.cols * EYE_SW);
		int heightY = cvRound(face.rows * EYE_SH);
		int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW));

		Mat topLeftOfFace = face(Rect(leftX, topY, widthX,heightY));
		Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));

		vector<Rect> lEyeR, rEyeR;

		lEyeDetector.detectMultiScale(topLeftOfFace, lEyeR, 1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH);
		lEyeDetector.detectMultiScale(topRightOfFace, rEyeR, 1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH);

		if(lEyeR.size() == 1 && rEyeR.size() == 1)
		{
			lEye = lEyeR[0];
			rEye = rEyeR[0];

			lEye.x += leftX;
			lEye.y += topY;

			rEye.x += rightX;
			rEye.y += topY;

			return true;
		}
	}

	return false;
}

//Alinea y recorta el rostro
void AlinearYRecortar(const Mat& face, Mat& warped, Rect leftEye, Rect rightEye)
{
	double DESIRED_LEFT_EYE_Y = 0.14;
    double DESIRED_LEFT_EYE_X = 0.19;

    int FaceWidth = 100;
    int FaceHeight = 100;

	Point left = Point(leftEye.x + leftEye.width/2, leftEye.y + leftEye.height/2);
	Point right = Point(rightEye.x + rightEye.width/2, rightEye.y + rightEye.height/2);
	Point2f eyesCenter = Point2f( (left.x + right.x) * 0.5f, (left.y + right.y) * 0.5f );

	// Recorta y alinea el ángulo entre los 2 ojos.
	double dy = (right.y - left.y);
	double dx = (right.x - left.x);
	double len = sqrt(dx*dx + dy*dy); //hipotenusa
	double angle = atan2(dy, dx) * 180.0 / CV_PI; //Angulo

    // Se recomienda alinear el rostro teniendo en cuenta el valor deseado
    //  del ojo izquierdo en el punto (0.19, 0.14) de una imagen escalada.
	const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);

	// Obtenemos la cantidad que debemos escalar para que la imagen sea del
	// tamaño deseado.
	double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * FaceWidth;
	double scale = desiredLen / len;

    // Relizamos la transformación matricial para rotar y alinear el rostro al
    // ángulo y tamaño deseado.
	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);

	// Cambiamos el centro de los ojos al deseado.
	rot_mat.at<double>(0, 2) += FaceWidth * 0.5f - eyesCenter.x;
	rot_mat.at<double>(1, 2) += FaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

    // Creamos una matriz "deformada" con las medidas deseadas
	warped = Mat(FaceHeight, FaceWidth, CV_8U, Scalar(128));

	warpAffine(face, warped, rot_mat, warped.size());
}

//Dibuja rectangulo y el nombre del rostro encontrado.
void MarcarYNombrar(Mat& dst, Rect rect, string msg, int LINE_WIDTH)
{
	Rect r = rect;
    rectangle(dst,Point(r.x, r.y),Point(r.x + r.width, r.y + r.height),CV_RGB(0,255,0), 2);

	int font = FONT_HERSHEY_DUPLEX;
	Size s = getTextSize(msg, font, 1, 1, 0);

	int x = (dst.cols - s.width) / 2;
	int y = rect.y + rect.height + s.height + 5;

	putText(dst, msg, Point(x, y), font, 1, CV_RGB(0,0,255), 1, CV_AA);
}

//Inicializa los clasificadores en cascada .xml y la webcam
bool Inicializar()
{
	if(!capture.open(0))
	{
		cout << "No se ha podido acceder a la webcam" << endl;
		return false;
	}

	if(!faceDetector.load("haarcascade_frontalface_alt_tree.xml"))
	{
		cout << "No se encuentra el archivo haarcascade_frontalface_alt_tree.xml" << endl;
		return false;
	}

	if(!lEyeDetector.load("haarcascade_eye_tree_eyeglasses.xml"))
	{
		cout << "No se encuentra el archivo haarcascade_eye_tree_eyeglasses.xml" << endl;
		return false;
	}

	if(!rEyeDetector.load("haarcascade_eye_tree_eyeglasses.xml"))
	{
		cout << "No se encuentra el archivo haarcascade_eye_tree_eyeglasses.xml" << endl;
		return false;
	}

	return true;
}

int main()
{
	Mat frame, copyFrame;
	Ptr <FaceRecognizer> model = createLBPHFaceRecognizer();
	vector<Mat> rostros;
	vector<int> ids;
	map<int , string> names;

	bool entrenado = false;
	bool agregarRostro = false;
	bool entrenar = false;
	int identificador = 0, capCount = 0;

    cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
    cout << "*            TRABAJO FINAL DE PROCESAMIENTO DIGITAL DE SENALES II:            *\n";
    cout << "*                                                                             *\n";
    cout << "*                           -RECONOCIMIENTO FACIAL-                           *\n";
    cout << "*                                                                             *\n";
    cout << "*                                                Alumno: Pablo Daniel Magallán*\n";
    cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
	string msg1 = "Seleccionar una de las siguientes opciones: \n\n\t[E] Iniciar Entrenamiento \n\t[ESC] Salir\n";
	string msg2 = "Estamos en la fase de entrenamiento, tomar al menos 5 capturas por favor: \n\n\t[A] Capturar Rostro \n\t[T] Finalizar Entrenamiento \n\t[ESC] Salir\n";
	cout << msg1;

	bool correct = Inicializar();

    FILE * pFile;
    pFile = fopen ("Registro.txt","w");

    fprintf (pFile, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    fprintf (pFile, "*            TRABAJO FINAL DE PROCESAMIENTO DIGITAL DE SENALES II:            *\n");
    fprintf (pFile, "*                                                                             *\n");
    fprintf (pFile, "*                           -RECONOCIMIENTO FACIAL-                           *\n");
    fprintf (pFile, "*                                                                             *\n");
    fprintf (pFile, "*                                                Alumno: Pablo Daniel Magallán*\n");
    fprintf (pFile, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");

	while (correct)
	{
		capture >> frame;

		//Reducir el tamaño de la imagen para mejor rendimiento, de modo que quede
		//con 480 pixeles de ancho. Reduce el tamaño en filas y columnas en la
		//misma proporcion (scale) para mantener la misma relacion fila-columna.
		float scale = frame.cols / (float) REDUCED_SIZE; //Columna escala

		if (frame.cols > REDUCED_SIZE) {
			int scaledHeight = cvRound(frame.rows / scale); //Fila escala redondeada
			resize(frame, frame, Size(REDUCED_SIZE, scaledHeight));
		}

        //Pasar imagen a escala de grises
		cvtColor(frame, copyFrame, CV_BGR2GRAY);

		//Ecualizar histograma
		equalizeHist(copyFrame, copyFrame);

		//Obtener las coordenadas del rostro y los ojos
		Rect face, lEye, rEye;

		if(EncontrarRostroYOjos(copyFrame, face, lEye, rEye))
		{
			//si el modo entrenamiento esta activo
			if(entrenado)
			{
				Mat nface;
				AlinearYRecortar(copyFrame(face), nface, lEye, rEye);

				//Agregar el rostro y su numero id a las correspondientes listas
				if(agregarRostro)
				{
					rostros.push_back(nface);
					ids.push_back(identificador);
					agregarRostro = false;

					capCount += 1;
					cout << "Se han capturado " << capCount << " Rostros" << endl;
				}

				//entrenar el modelo con los rostros capturados
				if(entrenar && rostros.size() >= 1)
				{
					model->update(rostros, ids);

					cout << "\nNombre de la persona: ";
					cin >> names[identificador];
					system("cls");

					entrenar = agregarRostro = entrenado = false;
					rostros.clear();
					ids.clear();
					identificador += 1;
					capCount = 0;

                    cout << "*******************************************************************************\n";
                    cout << "*            TRABAJO FINAL DE PROCESAMIENTO DIGITAL DE SENALES II:            *\n";
                    cout << "*                                                                             *\n";
                    cout << "*                           -RECONOCIMIENTO FACIAL-                           *\n";
                    cout << "*                                                                             *\n";
                    cout << "*                                                Alumno: Pablo Daniel Magallán*\n";
                    cout << "*******************************************************************************\n";
					cout << msg1;
				}
			}

            //Si ya se realizo el entrenamiento
			if(identificador >= 1)
			{
				int id = -1;
				double confidence = 0.0;

				Mat nface;
				AlinearYRecortar(copyFrame(face), nface, lEye, rEye);

                //threshold es el umbral de desicion del algoritmo
				//calquier confidence mayor que threshold id = -1
				//reducir o aumentar este valor segun nos convenga
				model->set("threshold", 70);
				model->predict(nface, id, confidence);

				if(id >= 0) //Si reconoció un rostro entrenado.
				{
                    string msg = "Hola " + names[id] + "!";

                    //Guardo en archivo registro fecha y hora que se
                    //encontró una persona
                    const char * c = msg.c_str();
                    time_t now = time(0);// current date/time
                    char* dt = ctime(&now);// convert to string form

                    fprintf (pFile,"Se encontro rostro: %s %s \n", dt, c);

					MarcarYNombrar(frame, face, msg , 20);
				}
				else MarcarYNombrar(frame, face, "???", 20);
			}
			else MarcarYNombrar(frame, face, "???", 20);
		}

		imshow("Reconocimiento de rostros", frame);

		switch (waitKey(30))
		{
		case 'T':
		case 't':
			entrenar = true;
			break;
		case 'A':
		case 'a':
			agregarRostro = entrenado;
			break;
		case 'E':
		case 'e':
			entrenado = true;
			system("cls");
			cout << msg2 << endl;
			break;
		case 27:
			return 0;
		}
	}
    fclose (pFile);
	system("pause");

	return 0;
}
