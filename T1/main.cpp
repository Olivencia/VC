/**
* Cristóbal Antonio Olivencia Carrión <cristobalolivencia@correo.ugr.es>
*
* VC - Visión por Computador
*
* Trabajo 1
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace cv;

/**
* Enumerados que se han usado para hacer más legible el código
*/
enum {LOW_FREQUENCY = 0, HIGH_FREQUENCY = 1, HYBRID_IMG = 2};
enum {ZEROS = 0, REFLECTION = 1};
enum {IMPLEMENTATION = 0, OPENCV = 1};

///////////////////////////////////////////////////////
///////////////////Funciones auxiliares////////////////
///////////////////////////////////////////////////////

/**
* Muestra las imágenes que se encuentran en el vector @images en una ventana llamada  
* como se establece en @window. Para cambiar de imagen hay que pulsar cualquier tecla.
*/
void showImages(vector<Mat> images, string window = "Ventana"){
	for(unsigned i=0; i<images.size(); i++){
		imshow(window, images[i]);
		waitKey(0);
	}
}

/**
* Muestra el canvas de las imágenes de baja y alta frecuencia junto con la resultante de
* juntar ambas, es decir, la imagen híbrida.
*/
void showHybridImagesCanvas(vector<vector<Mat> > imgs){
	vector<Mat> hybridImgs;
	hybridImgs.resize(imgs.size());

	for(unsigned i=0; i<imgs.size(); i++){
		hconcat(imgs[i][LOW_FREQUENCY], imgs[i][HIGH_FREQUENCY], hybridImgs[i]);
    	hconcat(hybridImgs[i], imgs[i][HYBRID_IMG], hybridImgs[i]);
	}
	showImages(hybridImgs);
}

/**
* Libera la memoria de las imágenes que están en el vector @images y a continuación 
* la memoria de dicho vector. 
* 
*/
void remove(vector<Mat> &images){
	for(unsigned i=0; i<images.size(); i++){
		images[i].release();
	}
	images.clear();
}

///////////////////////////////////////////////////////
//////////////////////Ejercicio 1//////////////////////
///////////////////////////////////////////////////////

/**
* En función del tamaño de la máscara @mask_size y el @type aplica unos bordes a la imagen @img para evitar
* problemas durante la convolución.
*/
Mat apply_edges(Mat img, int mask_size, int type = ZEROS){
	if(type == ZEROS){
		Mat horizontal_edge(img.rows, mask_size/2, img.type(), 0.0);
		hconcat(horizontal_edge, img, img); hconcat(img, horizontal_edge, img);

		Mat vertical_edge(mask_size/2, img.cols, img.type(), 0.0);
		vconcat(vertical_edge, img, img); vconcat(img, vertical_edge, img);
	}
	else if(type == REFLECTION){
		Mat horizontal_edge_left, horizontal_edge_right, vertical_edge_down, vertical_edge_up;

		horizontal_edge_left = img.rowRange(0, img.rows);
		horizontal_edge_left = horizontal_edge_left.colRange(0, mask_size/2);
		hconcat(horizontal_edge_left, img, img);			

		horizontal_edge_right = img.rowRange(0, img.rows);
		horizontal_edge_right = horizontal_edge_right.colRange(img.cols - mask_size/2, img.cols);
		hconcat(img,horizontal_edge_right, img);

		vertical_edge_up = img.rowRange(0, mask_size/2);
		vertical_edge_up = vertical_edge_up.colRange(0, img.cols);
		vconcat(vertical_edge_up, img, img);			

		vertical_edge_down = img.rowRange(img.rows - mask_size/2, img.rows);
		vertical_edge_down = vertical_edge_down.colRange(0, img.cols);
		vconcat(img,vertical_edge_down, img);
	}

	return img;
}

/**
* Obtiene el tamaño del kernel en función de @sigma y el @value son las unidades sigma que 
* se escogen de rango, por defecto 6 (-3,3)
*/
int getKernelSize(double sigma = 1.0, int value = 6){
	return round(value*sigma)+1;
}

/**
* Obtiene los valores de la máscara de convolución en funcion del parámetro @sigma y normaliza
* los valores obtenidos, todo esto se guarda en mask.
*/
Mat convMask(double sigma){
	int size = getKernelSize(sigma);
	double normal = 0, tmp_value;
	vector<double> v;
	v.resize(size);

	for(int i=1; i<=size/2; i++){
		tmp_value = exp(-0.5*((i*i)/(sigma*sigma)));

		v[(size/2)-i] = tmp_value;
		v[(size/2)+i] = tmp_value;
		normal += tmp_value;
	}
	normal*=2;
	tmp_value = exp(-0.5*(0/(sigma*sigma)));
	v[(size/2)] = tmp_value;
	normal += tmp_value;

	Mat mask(v, true);

	mask /= normal;
	
	return mask;
}

/**
* Guarda en @result el valor final que tendrá el pixel concreto del canal que se encuentre
* en el momento de llamar a esta función. El parámetro @signal es la señal que recibe para
* realizar la convolución con la máscara 1D @mask.
*/
void convMaskChannel(Mat &signal, Mat &mask, double &result){
	vector<double> aux(signal.cols);
	result = 0;
	for (int i = 0; i < signal.cols; i++){
		aux[i] = 0.0;
		for (int j = 0; j < signal.rows; j++){
			aux[i] += signal.at<double>(j, i)*mask.at<double>(j,0);
		}
		aux[i] *= mask.at<double>(i,0);
		result += aux[i];
	}
}

/**
* Para la imagen @img y con la máscara @mask se aplican los bordes a la imagen (para evitar
* problemas de desbordamiento), se coge una porción de la imagen y se llama a la función que
* aplica la convolución en función del canal que estemos.
*/
Mat gaussianConvFilter(Mat img, Mat &mask){
	int total_channels = img.channels();
	img.convertTo(img, CV_64F);
	Mat aux, img_edges = apply_edges(img, mask.rows, REFLECTION);
	vector<Mat> channels;
	for(int i=0; i < img.cols; i++){
		for(int j=0; j < img.rows; j++){
			aux = img_edges(Rect(i, j, mask.rows, mask.rows));

			if(aux.channels() > 1){
				split(aux, channels);
				for(int k=0; k<total_channels; k++) convMaskChannel(channels[k], mask, img.at<Vec3d>(j, i)[k]);
			}
			else convMaskChannel(aux, mask, img.at<double>(j, i));
		}
	}
	remove(channels);

	img.convertTo(img, CV_8UC3);
	return img;
}

/**
* Función general para el cálculo de la convolución de la imagen @img con el valor de @sigma,
* si no se pasa como parámetro valor a sigma, este tendrá por defecto el valor de 1.
*/
Mat gaussianFilter(Mat &img, double sigma = 1.0){
	Mat mask  = convMask(sigma);
	return gaussianConvFilter(img, mask);
}

///////////////////////////////////////////////////////
//////////////////////Ejercicio 2//////////////////////
///////////////////////////////////////////////////////

/**
* Devuelve la imagen resultante de la suma de todos los valores de la imagen @img y el valor establecido 
* en @value para todos los canales de dicha imagen.
*/
Mat sumValueToMat(Mat img, double value){
	Mat aux = img;
	int total_channels = aux.channels();
	for(int i=0; i < aux.cols; i++){
		for(int j=0; j < aux.rows; j++){
			if(total_channels > 1){
				for(int k=0; k < 3; k++){
					aux.at<Vec3d>(j, i)[k] += (-value);
				}
			}
			else 
			{
				aux.at<double>(j, i) += (-value);
			}
		}
	}
	return aux;
}

/**
* Aumenta la intensidad de los pixeles de todos los canales sumando el valor mínimo negativo existente 
* que se ha generado durante la obtención la imagen de alta frecuencia para mostrar de forma más clara
* en el canvas.
*/
void suppressNegativesValues(Mat &img){
	double min_value = 0, value;
	int total_channels = img.channels();
	for(int i=0; i < img.cols; i++){
		for(int j=0; j < img.rows; j++){
			if(total_channels > 1){
				for(int k=0; k < total_channels; k++){
					value = img.at<Vec3d>(j, i)[k];
					if (value < min_value) min_value = value;
				}
			}
			else 
			{
				value = img.at<double>(j, i);
				if (value < min_value) min_value = value;
			}
		}
	}
	if(min_value < 0) img = sumValueToMat(img, min_value);
}

/**
* Obtiene una imagen híbrida formada por la convolución de la imagen @img1 y la imagen @img2
* donde en la primera imagen se pretende eliminar las frecuencias altas de la imagen mientras 
* que en la imagen 2 si obtener las frecuencias altas. La variable @mode indica que funcion de
* normalización de imágenes usar. 
*/
vector<Mat> hybrid_images(Mat img1, Mat img2, double sigma1, double sigma2, int mode = IMPLEMENTATION
	)
{
    Mat high_freq, high_freq_brigthness, low_freq, img1_smooth, img2_smooth, aux;
    vector<Mat> hybrid_image;

    resize(img2, img2, Size(img1.cols, img1.rows));
    img1_smooth = gaussianFilter(img1, sigma1);
    img2_smooth = gaussianFilter(img2, sigma2);
    img2_smooth.convertTo(img2_smooth, CV_64F);
    img1_smooth.convertTo(img1_smooth, CV_64F);

    img1.convertTo(img1, CV_64F);
    img2.convertTo(img2, CV_64F);

    low_freq = img2_smooth;
    high_freq = img1 - img1_smooth;

    if(mode == IMPLEMENTATION) suppressNegativesValues(high_freq);
    else if(mode == OPENCV) normalize(high_freq, high_freq, 0, 255, NORM_MINMAX, CV_64F);

    aux = (low_freq + high_freq)/2;

    low_freq.convertTo(low_freq, CV_8UC3);
    high_freq.convertTo(high_freq, CV_8UC3);
    aux.convertTo(aux, CV_8UC3);
    
    hybrid_image.push_back(low_freq);
    hybrid_image.push_back(high_freq);
    hybrid_image.push_back(aux);
	
	return hybrid_image;
}

///////////////////////////////////////////////////////
//////////////////////Ejercicio 3//////////////////////
///////////////////////////////////////////////////////

/**
* Dada una imagen @img devuelve la misma con un tamaño 1/4 más pequeño
*/

Mat subsampling(Mat img){
	int total_channels = img.channels();
	Mat aux(img.rows/2, img.cols/2, img.type());
	img.convertTo(img, CV_64F);
	aux.convertTo(aux, CV_64F);
	for(int i=0; i < aux.cols; i++){
		for(int j=0; j < aux.rows; j++){
			if(total_channels > 1){
				for(int k=0; k < total_channels; k++){
					aux.at<Vec3d>(j,i)[k] = img.at<Vec3d>(j*2, i*2)[k];
				}
			}
			else aux.at<double>(j,i) = img.at<double>(j*2, i*2);
		}
	}
	aux.convertTo(aux, CV_8UC3);
	return aux;
}

/**
* Función que reemplaza a la de OpenCV pyrDown(). Primero genera la imagen @img alisada con sigma=1
* y a ésta le realiza un subsampling guardandolo de nuevo en img.
*/

void pyramidDown(Mat &img){
	Mat img_smooth = gaussianFilter(img);
	img = subsampling(img_smooth);
}

/**
* Dada una imagen híbrida @img y el número de niveles @levels obtener la pirámide Gaussiana 
* de dicha imagen con los niveles dados. Se trata de un submuestreo y concatenación de imágenes. 
*/
Mat gaussianPyramid(Mat img, int levels){
	Mat pyramid = img, aux = img, tmp;
	for (int i = 0; i<levels; i++){
		pyramidDown(aux);
		tmp = Mat::zeros(img.rows - aux.rows, aux.cols, aux.type());
		vconcat(aux, tmp, tmp);
		hconcat(pyramid, tmp, pyramid);
	}
	return pyramid;
}

int main() {

	//Imágenes disponibles
	Mat plane = imread("imagenes/plane.bmp");
	Mat bird = imread("imagenes/bird.bmp");
	Mat cat = imread("imagenes/cat.bmp");
	Mat dog = imread("imagenes/dog.bmp");
	Mat bike = imread("imagenes/bicycle.bmp");
	Mat moto = imread("imagenes/motorcycle.bmp");
	Mat einstein = imread("imagenes/einstein.bmp");
	Mat marilyn = imread("imagenes/marilyn.bmp");
	Mat fish = imread("imagenes/fish.bmp");
	Mat submarine = imread("imagenes/submarine.bmp");

	///////////////////////////////////////////////////////
	//////////////////////Ejercicio 1//////////////////////
	///////////////////////////////////////////////////////

	vector<Mat> convImgs;

	cout << "Para cambiar entre imágenes pulse cualquier letra." << endl;

	cout << "//////////////////////Ejercicio 1//////////////////////" << endl;
	cout << "Comenzando la convolución para las diferentes imágenes..." << endl;
	convImgs.push_back(gaussianFilter(bird,2));
	convImgs.push_back(gaussianFilter(dog,4));
	convImgs.push_back(gaussianFilter(bike,5));
	convImgs.push_back(gaussianFilter(einstein,2));
	convImgs.push_back(gaussianFilter(fish,4));
	cout << "Convolución finalizada." << endl;

	showImages(convImgs);
	remove(convImgs);

	///////////////////////////////////////////////////////
	//////////////////////Ejercicio 2//////////////////////
	///////////////////////////////////////////////////////

	vector<vector<Mat> > hybridImgs;

	cout << "//////////////////////Ejercicio 2//////////////////////" << endl;
	cout << "Comenzando el cáculo de las imágenes híbridas." << endl;
	cout << "Esto puede tardar, espere por favor..." << endl;
	hybridImgs.push_back(hybrid_images(plane, bird, 3, 5, IMPLEMENTATION));
	hybridImgs.push_back(hybrid_images(cat, dog, 6, 7, IMPLEMENTATION));
	hybridImgs.push_back(hybrid_images(bike, moto, 2, 7, IMPLEMENTATION));
	hybridImgs.push_back(hybrid_images(marilyn, einstein, 3 , 4, IMPLEMENTATION));
	hybridImgs.push_back(hybrid_images(fish, submarine, 3, 4, IMPLEMENTATION));
	cout << "Cálculo de las imágenes híbridas finalizado." << endl;

	showHybridImagesCanvas(hybridImgs);

	///////////////////////////////////////////////////////
	//////////////////////Ejercicio 3//////////////////////
	///////////////////////////////////////////////////////

	vector<Mat> pyramidImgs;

	cout << "//////////////////////Ejercicio 3//////////////////////" << endl;
	cout << "Comenzando la representación de las pirámides Gaussianas..." << endl;
	for(unsigned i=0; i<hybridImgs.size(); i++){
		pyramidImgs.push_back(gaussianPyramid(hybridImgs[i][HYBRID_IMG], 5));
	}
	cout << "Obtención de las diferentes pirámides realizada." << endl;
	
	showImages(pyramidImgs);
	for(unsigned i=0; i<hybridImgs.size(); i++){
		remove(hybridImgs[i]);
	}
	remove(pyramidImgs);

	return 0;
}
