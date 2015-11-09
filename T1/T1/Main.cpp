/**
* Cristóbal Antonio Olivencia Carrión <cristobalolivencia@correo.ugr.es>
*
* VC - Visión por Computador
*
* Trabajo 1
*/



#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/**
* Muestra la ventana con el nombre @nom y la imagen @img que se le pasa como argumento
*/
void muestraVentana(Mat &img, string nom = "Ventana"){
	imshow(nom, img);
	waitKey(0);
	destroyWindow(nom);
}

/**
* Muestra una ventana con las imagenes @img proporcionadas y el nombre de la ventana @nom
*/
void dibujarImagenes(vector<Mat> &img, string nom = "Ventana"){
	Mat tmp;
	int fil = img[0].rows, col = img[0].cols, g = 1, tam = img.size();
	Size s;
	s.width = col;
	s.height = fil;

	if (tam>3){
		if (tam%2 == 0) g = 2;
		else if (tam%3 == 0) g = 3;
		else
		{
			g = 2;
			tam++;
			Mat b(fil, col, 16, Scalar(255, 255, 255));
			img.push_back(b);
		}
	}

	vector<Mat> aux(g);

	for (int i = 0; i<g; i++){
		img[i*(tam / g)].convertTo(aux[i], CV_8UC3);
		resize(aux[i], aux[i], s);
		for (int j = i*(tam / g) + 1; j<tam / g + tam / g*i; j++){
			img[j].convertTo(tmp, CV_8UC3);
			resize(tmp, tmp, s);
			hconcat(aux[i], tmp, aux[i]);
		}
	}

	Mat res = aux[0];
	for (int i = 1; i<g; i++){
		vconcat(res, aux[i], res);
	}
	muestraVentana(res, nom);
}


/**
* Obtiene el valor de la máscara a partir de x y sigma
*/
double valorGaussiano(int x, double sigma){
	return exp(-0.5*((x*x)/(sigma*sigma)));
}

/**
* Genera una ventana en la que pinta la imagen que se pasa en img.
*/
Mat convMask(int tam, double sigma){
	double normal = 0;
	vector<double> m;
	m.resize(tam);
	
	for(int i = 0; i <= tam/2; i++){
		normal += m[(tam/2)-i] = m[(tam/2)+i] = valorGaussiano(i, sigma);
	}
	
	normal = 1.0/(normal*2-valorGaussiano(0, sigma));

	Mat mask(m,true);
	mask *= normal;
	return mask;

}

/**
* Se añaden bordes con valor de pixel 0 
*/
Mat bordes(Mat img, Mat & mask){
	Mat b1(img.rows, mask.rows/2, img.type(), 0.0);
	hconcat(b1, img, img);
	hconcat(img, b1, img);

	Mat b2(mask.rows/2, img.cols, img.type(), 0.0);
	vconcat(b2, img, img);
	vconcat(img, b2, img);
	return img;
}


/**
* Calcula la convolución de una matriz @senal de un canal con la máscara mask y lo almacena en res
*/
void convMask1Canal(Mat &senal, Mat &mask, double &res){
	vector<double> aux(senal.cols);
	res = 0.0;
	for (int i = 0; i<senal.cols; i++){
		aux[i] = 0.0;
		for (int j = 0; j<senal.rows; j++){
			aux[i] += senal.at<double>(j, i)*mask.at<double>(j, 0);
		}
		aux[i] *= mask.at<double>(i, 0);
		res += aux[i];
	}
}


/**
* Calcula la convolución de una matriz  @senal de tres canales con la máscara mask y lo almacena en res
*/
void convMask3Canal(Mat &senal, Mat &mask, Vec3d &res){
	vector<Mat> canales;
	split(senal, canales);
	for (int i = 0; i<senal.channels(); i++) convMask1Canal(canales[i], mask, res[i]);
}


/**
* Obtiene la convolución de la imagen @src con @mask
*/
Mat filtroGaussiano(Mat img, Mat &mask){
	img.convertTo(img, CV_64F);
	Mat aux;
	Mat b = bordes(img, mask);
	for (int i = 0; i < img.cols; i++){
		for (int j = 0; j < img.rows; j++){
			aux = b(Rect(i, j, mask.rows, mask.rows));
			if (aux.channels() > 1){			
				convMask3Canal(aux, mask, img.at<Vec3d>(j, i));
			}			
			else
				convMask1Canal(aux, mask, img.at<double>(j, i));
		}
	}
	return img;
}


/**
* Obtiene la convolución de la imagen @src con @sigma
*/
Mat filtroGaussiano(Mat &img, double sigma = 1.0)
{
	Mat mask = convMask(4*sigma+1, sigma), im = filtroGaussiano(img, mask);
	return im;
}

/**
* Obtiene una imagen híbrida de dos imágenes de alta y baja frecuencia.
*/
Mat imagenHibrida(Mat img1, Mat img2, double sigma1, double sigma2){
	Mat a, b, h, f1, f2, tmp;
	int f = img1.rows, c = img1.cols;
	resize(img2, img2, Size(c,f));
	f1 = filtroGaussiano(img1, sigma1);
	f1.convertTo(f1, CV_64F);
	f2 = filtroGaussiano(img2, sigma2);
	f2.convertTo(f2, CV_64F);
	
	img1.convertTo(img1, CV_64F);
	img2.convertTo(img2, CV_64F);
	
	a = img2 - f2;
	b = f1;

	if (sigma1 < sigma2){
		a = img1 - f1;
		b = f2;
	}

	h = a + b;
	hconcat(b, a, tmp);
	hconcat(tmp, h, tmp);
	return tmp;
}

/**
* Obtiene una pirámide Gaussiana de los niveles indicados en @levels para la imagen @src
*/
Mat piramideGaussiana(Mat &img, int levels)
{
	Mat piramide = img,  aux = piramide;
	Mat tmp;
	for (int i = 0; i<levels; i++)
	{
		pyrDown(aux, aux);
		tmp = Mat::zeros(aux.rows, img.cols - aux.cols, aux.type());
		hconcat(tmp, aux, tmp);
		vconcat(piramide, tmp, piramide);
	}
	return piramide;
}

/**
* Devuelve una imagen en negro con bordes en escala de grises (Canny)
*/
Mat Dg_Canny(Mat img, int lowThreshold, int ratio, int kernel_size){
	Mat res, dg;
	cvtColor(img, img, CV_BGR2GRAY);
	blur(img, dg, Size(3, 3));
	Canny(dg, dg, lowThreshold, lowThreshold*ratio, kernel_size);
	res = Scalar::all(0);
	img.copyTo(res, dg);
	return res;
}
/**
* Igual que la función Dg_Canny salvo que en este caso el fondo es blanco
*/
Mat Canny_blanco(Mat img){
	int pixel;
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			pixel = img.data[img.channels()*(img.cols*i + j)];

			if (pixel == 0){
				img.data[img.channels()*(img.cols*i + j)] = 255;
			}
		}
	}

	return img;
}

/**
* Resalta los pixels en función del fondo, si es negro los bordes lo más blanco posible y viceversa
*/
Mat Canny_resaltado(Mat img, int tipo = 0){
	int pixel;
	if (tipo == 0){
		for (int i = 0; i < img.rows; i++){
			for (int j = 0; j < img.cols; j++){
				pixel = img.data[img.channels()*(img.cols*i + j)];

				if (pixel > 0){
					img.data[img.channels()*(img.cols*i + j)] = 255;
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < img.rows; i++){
			for (int j = 0; j < img.cols; j++){
				pixel = img.data[img.channels()*(img.cols*i + j)];

				if (pixel < 255){
					img.data[img.channels()*(img.cols*i + j)] = 0;
				}
			}
		}
	}

	return img;
}

int main(int argc, char* argv[])
{
	//Imágenes disponibles
	Mat lena = imread("Imagenes/lena.jpg", 1);
	Mat cat = imread("Imagenes/cat.bmp", 1);
	Mat plane = imread("Imagenes/plane.bmp", 1);
	Mat bird = imread("Imagenes/bird.bmp", 1);
	Mat dog = imread("Imagenes/dog.bmp", 1);
	Mat einstein = imread("Imagenes/einstein.bmp", 1);
	Mat bici = imread("Imagenes/bicycle.bmp");
	Mat moto = imread("Imagenes/motorcycle.bmp");
	Mat marilyn = imread("Imagenes/marilyn.bmp");
	Mat fish = imread("Imagenes/fish.bmp");
	Mat submarine = imread("Imagenes/submarine.bmp");


	vector<Mat> imagenes, m, im(imagenes.size());

	//>Imágenes usadas para la convolución
	imagenes.push_back(lena);
	imagenes.push_back(cat);
	imagenes.push_back(plane);
	imagenes.push_back(bird);
	imagenes.push_back(dog);
	imagenes.push_back(einstein);

	vector<pair<Mat, Mat>> h;
	pair<Mat,Mat> aux;
	pair<string, int> a;

	//Imágenes seleccionadas para hibridas y piramide
	aux.first = submarine; aux.second = fish;
	h.push_back(aux);
	aux.first = dog; aux.second = cat;
	h.push_back(aux);
	aux.first = marilyn; aux.second = einstein;
	h.push_back(aux);
	aux.first = bird; aux.second = plane;
	h.push_back(aux);
	aux.first = moto; aux.second = bici;
	h.push_back(aux);


	//////////////////////////////////////////////////////EJERCICIO 1////////////////////////////////////////////////
	cout << "*************************************************" << endl;
	cout << "Ejercicio 1" << endl;
	cout << "Convolucion 2D de una imagen" << endl;
	cout << "*************************************************" << endl;
	
	for (unsigned int i = 0; i<imagenes.size(); i++){
		m.push_back(filtroGaussiano(imagenes[i], 1.0));
		m.push_back(filtroGaussiano(imagenes[i], 5.0));
	}	
	dibujarImagenes(m,"Convolucion");
	m.clear();
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////EJERCICIO 2////////////////////////////////////////////////
	vector<Mat> hibridas(h.size());
	cout << "*************************************************" << endl;
	cout << "Ejercicio 2" << endl;
	cout << "Imagenes hibridas" << endl;
	cout << "*************************************************" << endl;
	for (unsigned int i = 0; i < h.size(); i++){
		hibridas[i] = imagenHibrida(h[i].first, h[i].second, 2, 9);

		m.push_back(hibridas[i]);

		dibujarImagenes(m, "Imagenes hibridas");
		m.clear();
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////EJERCICIO 3////////////////////////////////////////////////
	cout << "*************************************************" << endl;
	cout << "Ejercicio 3" << endl;
	cout << "Piramide Gaussiana" << endl;
	cout << "*************************************************" << endl;
	for (unsigned int i = 0; i<hibridas.size(); i++){
			muestraVentana(piramideGaussiana(hibridas[i], 5),"Piramide Gaussiana");
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////BONUS.1////////////////////////////////////////////////
	cout << "*************************************************" << endl;
	cout << "Bonus.1" << endl;
	cout << "Deteccion de bordes y pintar las lineas de un color mas visible" << endl;
	cout << "Izquierda: sin resaltar bordes Derecha: bordes resaltados" << endl;
	cout << "*************************************************" << endl;
	//Aquí obtendremos los bordes de una imagen (foto de la izquierda) y a continuación los resaltamos en otra foto
	//(foto de la derecha) de forma que sean más visible los bordes
	int threshold = 40, ratio = 3, kernel_size = 3;
	Mat img_canny = Dg_Canny(moto, threshold, ratio, kernel_size), canny1, canny2,canny3;
	img_canny.copyTo(canny1);
	img_canny.copyTo(canny2);
	Mat img_canny_blanco =  Canny_blanco(img_canny);
	img_canny_blanco.copyTo(canny3);
	Mat img_canny_resaltado_blanco = Canny_resaltado(canny2);
	Mat img_canny_resaltado_negro = Canny_resaltado(canny3,1);
	vector<Mat> c;
	c.push_back(canny1);
	c.push_back(img_canny_resaltado_blanco);
	c.push_back(img_canny_blanco);
	c.push_back(img_canny_resaltado_negro);
	
	dibujarImagenes(c);
	return 0;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}
