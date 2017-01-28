/**
* Cristóbal Antonio Olivencia Carrión <cristobalolivencia@correo.ugr.es>
*
* VC - Visión por Computador
*
* Trabajo 3
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <time.h>

using namespace std;
using namespace cv;

/**
* Enumerados que se han usado para hacer más legible el código
*/
enum {ZEROS = 0, REFLECTION = 1};
enum {ONLY_CORRESPONDENCES = 1, ONLY_PANORAMA = 2, ONLY_EPIPOLAR = 3, ONLY_FUNDAMENTAL_MAT = 4, ALL = 5};
enum {TYPE_AKAZE = 0, TYPE_KAZE = 1, TYPE_ORB = 2, TYPE_BRISK = 3};
enum {NO_DIST = 0, RAD_DIST = 1, TANG_DIST = 2, TANG_AND_RAD_DIST = 3};

/////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////Funciones auxiliares///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

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

/**
* En función del tamaño de la máscara @mask_size y el @type aplica unos bordes a la imagen @img para evitar
* problemas durante la convolución.
*/
Mat apply_edges(Mat img, int mask_size, int type = ZEROS){
	if(type == ZEROS){
		Mat horizontal_edge(img.rows, mask_size/2, img.type(), 0.0);
		hconcat(horizontal_edge, img, img); 
		hconcat(img, horizontal_edge, img);

		Mat vertical_edge(mask_size/2, img.cols, img.type(), 0.0);
		vconcat(vertical_edge, img, img); 
		vconcat(img, vertical_edge, img);
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
* Comprobación de si el pixel con coordenadas x e y como @central es un máximo local de la matriz de
entorno @matrix. Devuelve true en caso afirmativo, false en caso contrario.
*/
bool isLocalMax(Mat matrix, int central){
	bool is_max = true;
	float max = matrix.at<uchar>(central,central);
	for(int i=0; i < matrix.cols; i++){
		for(int j=0; j < matrix.rows; j++){
			if(max < matrix.at<uchar>(j,i)) is_max = false;
		}
	}
	return is_max;
}

/**
* Dada una imagen @img, eliminar los bordes que contienen negros para hacer más visible y
menos pesada la imagen.
*/
void omitEdges(Mat &img){
	Mat img_gray;
  	cvtColor(img, img_gray, COLOR_BGR2GRAY);
  	img_gray.convertTo(img_gray, CV_8UC1);
	int cnt = 0, down_row = 0, up_row = img_gray.rows-1, right_col, left_col;
	for(int i = 0; i < img_gray.cols; i++){	
		for(int j = 0; j < img_gray.rows; j++){
			int value = (int) img_gray.at<uchar>(j, i);
			if( value > 0){
				if(cnt == 0){
					left_col = i;
					cnt++;
				}
				if(down_row < j) down_row = j; 

				if(up_row > j)	up_row = j;

				right_col = i;
			}
		}
	}
	img = img(Rect(left_col, up_row, right_col-left_col+1, down_row-up_row+1));
}

/**
* Devuelve el tipo de imagen con el string correspondiente para que sea entendible.
*/
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

/////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////Funciones de la Práctica 2////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

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
* Obtención de la matriz con valores potenciales de puntos harris mediante la fórmula 
Det(@harris) - k*Traza(@harris)^2 siendo @harris la matriz de datos obtenidos anteriormente con 
la llamada a la funcion cornerEigenValsAndVecs()
*/
void harrisDetector(Mat &harris, float k){
	float l1, l2, value;
	Mat aux = Mat::zeros(harris.size(), CV_32FC1);
	for( int i = 0; i < harris.cols; i++ ){ 
	 	for( int j = 0; j < harris.rows; j++ ){
	      	l1 = harris.at<Vec6f>(j,i)[0];
	      	l2 = harris.at<Vec6f>(j,i)[1];
	      	value =  l1*l2 - k*pow((l1+l2),2);
	      	aux.at<float>(j,i) = value;
	  }
	}	
  harris = aux;
}


/**
* Dada la matriz de entorno @matrix y las coordenadas del punto central de este, almacenar en la
imagen binaria el valor de 255 y el resto 0 si dicho valor en las coordenadas @x e @y es el máximo
local, en caso contrario ponerlo a 0.
*/
void applyLocalMax(Mat matrix, Mat &res, int x, int y){
	int central = matrix.rows/2;
	if(isLocalMax(matrix, central)){
		for(int i=0; i < matrix.cols; i++){
			for(int j=0; j < matrix.rows; j++){
				res.at<uchar>(j+x,i+y) = 0;
			}
		}
		res.at<uchar>(x+central,y+central) = 255;
	}
	else res.at<uchar>(x+central,y+central) = 0;
}

/**
* Supresión de los no máximos para quedarnos con los puntos más relevantes de @matrix en función del
tamaño del entorno y la obtención de la imagen binaria con los puntos buenos.
*/
Mat supressNoMax(Mat &matrix, int window_size){
	Mat aux, res(matrix.size(), CV_8UC1, 255);
	res.col(res.cols-1) = 0;
	res.row(res.rows-1) = 0;
	res.col(0) = 0;
	res.row(0) = 0;
	int cnt = 0;
	for(int i=0; i <= matrix.cols - window_size; i++){
		for(int j=0; j <= matrix.rows - window_size; j++){	
			if(res.at<uchar>(j+window_size/2, i+window_size/2) == 255){
				aux = matrix(Rect(i, j, window_size, window_size));
				applyLocalMax(aux, res, j, i);
				cnt++;
			}
		}
	}
	return res;
}

/**
* Obtención de la estructura de datos usada para tratar los puntos harris(coordenadas, valor, escala)
de @matrix.
*/
vector<Mat> getHarrisData(Mat matrix, Mat binary, int level){
	vector<Mat> data;
	float value;
	int cnt = 0;
	Mat harrisPoints, harrisX, harrisY, scale;
	for(int i = 0; i < matrix.cols; i++){
  		for(int j = 0; j < matrix.rows; j++){
  			if(binary.at<uchar>(j, i) == 255){
	  			value = matrix.at<float>(j, i);
				harrisPoints.push_back(value);
	  			harrisX.push_back(j);
	  			harrisY.push_back(i);
	  			scale.push_back(1/pow(2,level));
	  			cnt++;
  			}
		}
  	}
  	data.push_back(harrisPoints);
  	data.push_back(harrisX);
  	data.push_back(harrisY);
  	data.push_back(scale);
  	return data;
}

/**
* Obtención de los mejores puntos @corners y cantidad @totalPoints, tras ordenar de mayor a menor los
puntos Harris almacenados en @matrix. 
*/
void getBestPoints(vector<Mat> matrix, Mat img_gray, vector<Point2f> &corners, int &totalPoints){
	Mat aux;
	float harrisX, harrisY;
	Size winSize = Size( 5, 5 ), zeroZone = Size( -1, -1 );
	TermCriteria criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );

	sortIdx(matrix[0],aux,CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

	for(int i = 0; i < matrix[0].rows && matrix[0].at<float>(aux.at<int>(i,0),0) > 0; i++){
		harrisX = matrix[1].at<int>(aux.at<int>(i,0),0);
		harrisY = matrix[2].at<int>(aux.at<int>(i,0),0);
        Point2f point(harrisY, harrisX);
		corners.push_back(point);
		totalPoints++;
	}

	cornerSubPix(img_gray, corners, winSize, zeroZone, criteria );
}

/**
* Generación de los círculos y dirección del gradiente de la imagen @original teniendo en cuenta el
radio @radius, nivel @level y la estructura de datos de los puntos Harris (x, y, valor) en @matrix
*/
void drawCircleAndOrientation(vector<Mat> &matrix, Mat &original, Mat img_gray, int radius, int level){
	Mat aux, gx, gy, angles, mag;
	int totalPoints = 0;
	float counterMax, harrisX, harrisY, harrisdX, harrisdY;
	vector<Point2f> corners;

	switch(level){
		case 0: counterMax = 0.7;
		break;
		case 1: counterMax = 0.2;
		break;
		case 2: counterMax = 0.1;
		break;
	} 

	getBestPoints(matrix, img_gray, corners, totalPoints);

    Sobel(img_gray, gx, CV_32FC1, 1, 0);
    Sobel(img_gray, gy, CV_32FC1, 0, 1);
    phase(gx, gy, angles);

	for(int i = 0; i < 1500*counterMax && i < totalPoints ; i++){
		harrisX=corners[i].y;
		harrisY=corners[i].x;
		
		float ang = angles.at<float>(harrisX,harrisY);
	    float dx=radius*cos(ang);
	    float dy=radius*sin(ang);

	    harrisdX = ( corners[i].y ) * pow(2,level) + dx;
	    harrisdY = ( corners[i].x ) * pow(2,level) + dy;

		circle(original, Point2f(harrisY*pow(2,level), harrisX*pow(2,level)), radius, Scalar(0,0,255), 1, CV_AA , 0);
	    line(original, Point2f(harrisY*pow(2,level),harrisX*pow(2,level)),Point2f(harrisdY, harrisdX),Scalar(0,0,255), 1, CV_AA);
	}
	imshow("Puntos Harris", original);
	waitKey(0);
}

/**
* Dada una imagen @img y el número de reducciones @reductions obtener los circulos y líneas 
de los puntos Harris de dicha imagen con las reducciones dadas. 
*/
Mat scalePyramid(Mat img, int reductions){
	Mat aux = gaussianFilter(img);
	float k = 0.04, scale;
	int bsize = 3, aperture = 3, wsize = 3, cradius;
	for (int i = 0; i<=reductions; i++){
		Mat img_gray, harris, binary;
		cradius = (wsize+1)*(i+1);
		scale = 1/(i+1)*pow(2,i);

		cvtColor(aux, img_gray, COLOR_BGR2GRAY);

	  	cornerEigenValsAndVecs( img_gray, harris, bsize, aperture, BORDER_DEFAULT );

	  	harrisDetector(harris, k);
	  	
	  	binary = supressNoMax(harris, wsize);

	  	vector<Mat> harrisData = getHarrisData(harris, binary, scale);

	  	drawCircleAndOrientation(harrisData, img, img_gray, cradius, i);
	  	remove(harrisData);
		pyramidDown(aux);
	}
	return img;
}

/**
* Con la lista de puntos @p1 y @p2 correspondientes a los keypoints de las images @img1 e @img2 
respectivamente encontrar su homografía y guardar en @mosaic la imagen resultante.
*/
void blendImages(Mat img1, Mat img2, Mat &res, vector<Point2f> p1, vector<Point2f> p2){
	Mat homography1, homography2, homography_final;

	res = Mat::zeros(2*(img1.rows+img2.rows), 2*(img1.cols+img2.cols), img1.type());
	
	homography1 = findHomography(p2, p1, CV_RANSAC, 1);
	homography1.convertTo(homography1, CV_32FC1);
	homography2 = Mat::eye(3, 3, CV_32FC1);
	homography2.at<float>(Point(2,0)) = (img1.cols+img2.cols)/2;
	homography2.at<float>(Point(2,1)) = 100;

	homography_final = homography2 * homography1;
	warpPerspective(img1, res, homography2, Size(res.cols, res.rows), INTER_LINEAR, BORDER_TRANSPARENT);
	warpPerspective(img2, res, homography_final, Size(res.cols, res.rows), INTER_LINEAR, BORDER_TRANSPARENT);

	omitEdges(res);
}



/////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////Funciones de la Práctica 3////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////Ejercicio 1/////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/**
* Dada la matriz @matrix realiza el determinante de sus valores
*/
int det(Mat matrix){
	return round(determinant(matrix));
}

/**
* Devuelve una matriz aleatoria 3x4 con determinante distinta de 0
*/
Mat generateRandomCamera(){
	cv::theRNG().state = time(NULL);
	Mat matrix = Mat(3, 4, CV_32FC1), aux;
    do{
    	randu(matrix, Scalar::all(0), Scalar::all(255));
		aux = matrix(Rect(0, 0, 3, 3));
	}
    while(determinant(aux) == 0);
	return matrix;
}

/**
* Genera el patrón de puntos del mundo 3D correspondientes con 0:0.1:0.1 - 0.1:0.1:0
*/
vector<Point3f> pointPattern(){
	vector<Point3f> pattern;
	for (double k1= 0.1; k1 <= 1; k1 += 0.1) {
        for (double k2 = 0.1; k2 <= 1; k2 += 0.1) {
            pattern.push_back(Point3f( 0, k1, k2));
            pattern.push_back(Point3f(k2, k1, 0));
        }
    }
	return pattern;
}

/**
* Dado un punto pasarlo a coordenadas homogéneas
*/
Mat toHomogeneous(Point3f point){
	Mat p = Mat(4,1,CV_32FC1);
	p.at<float>(Point(0,0)) = point.x; p.at<float>(Point(0,1)) = point.y;
	p.at<float>(Point(0,2)) = point.z; p.at<float>(Point(0,3)) = 1;
	return p;
}

/**
* Dado un punto pasarlo a 2D
*/
Point2f to2Dpoint(Mat point){
	Point2f aux;
	float z = point.at<float>(Point(0,2));
	aux.x = point.at<float>(Point(0,0))/z;
	aux.y = point.at<float>(Point(0,1))/z;
	return aux;
}

/**
* Dada la matriz cámara @P y el vector de los puntos 3D del mundo @points obtener los puntos 2D resultantes de multiplicar los puntos 3D por la matriz de la cámara 
*/
vector<Point2f> projectPoints(Mat P, vector<Point3f> points) {
	vector<Mat> projection;
	vector<Point2f> ppoints;

	vector<Mat> homogeneousPoints;
	for(unsigned i=0; i<points.size(); i++)
		homogeneousPoints.push_back(toHomogeneous(points[i]));

	for(unsigned i=0; i<homogeneousPoints.size(); i++)
		projection.push_back(P*homogeneousPoints[i]);

	for(unsigned i=0; i<projection.size(); i++)
		ppoints.push_back(to2Dpoint(projection[i]));

	return ppoints;
}

/**
* Los puntos @points son puntos 2D resultantes de la proyección en coordenadas del mundos y @rows y @cols las filas y columnas respectivas a la imagen en la que se va a pintar. Aquí se obtendrán los puntos en una "escala" mayor para que puedan verse correctamente al pintarlos.
*/
vector<Point2f> resizePoints(vector<Point2f> points, Size size){
	vector<Point2f> pixelProjection;
	Point2f pixelPoint;
	for(unsigned i=0; i<points.size(); i++){
		pixelPoint = Point2f((size.width/2)*points[i].x, (size.height/2)*points[i].y);
		pixelProjection.push_back(pixelPoint);
	}
	return pixelProjection;
}

/**
* Una vez que tenemos los puntos en correspondencias 2D 3D obtenemos la matriz de coeficientes. 
*/
Mat estimateMatrix(vector<Point2f> points2d, vector<Point3f> points3d){
	Mat aux = Mat::zeros(points2d.size()*2, 12, CV_32FC1);

	for(unsigned i=0; i<points2d.size(); i++){
		aux.at<float>(Point(0,2*i)) = aux.at<float>(Point(4,2*i+1)) = points3d[i].x;
		aux.at<float>(Point(1,2*i)) = aux.at<float>(Point(5,2*i+1)) = points3d[i].y;
		aux.at<float>(Point(2,2*i)) = aux.at<float>(Point(6,2*i+1)) = points3d[i].z;
		aux.at<float>(Point(3,2*i)) = aux.at<float>(Point(7,2*i+1)) = 1;

		aux.at<float>(Point(8,2*i)) = -points2d[i].x*points3d[i].x;
		aux.at<float>(Point(9,2*i)) = -points2d[i].x*points3d[i].y;
		aux.at<float>(Point(10,2*i)) = -points2d[i].x*points3d[i].z;
		aux.at<float>(Point(11,2*i)) = -points2d[i].x;
		
		aux.at<float>(Point(8,2*i+1)) = -points2d[i].y*points3d[i].x;
		aux.at<float>(Point(9,2*i+1)) = -points2d[i].y*points3d[i].y;
		aux.at<float>(Point(10,2*i+1)) = -points2d[i].y*points3d[i].z;
		aux.at<float>(Point(11,2*i+1)) = -points2d[i].y;
	}
	return aux;
}

/**
* Dados los puntos 2D 3D obtener la matriz de coeficientes y mediante la descomposición en valores singulares, tras generar Vt (V traspuesta) con la función compute obtener la última fila y almacenarlo en la matriz de la cámara. 
*/
Mat obtainMatrixCamera(vector<Point2f> points2d, vector<Point3f> points3d){
	Mat coefs = estimateMatrix(points2d, points3d);
	Mat camera = Mat(3, 4, CV_32FC1);
	Mat w, u, vt;

	SVD::compute(coefs, w, u, vt);

	for(int i=0; i<12; i++){
		camera.at<float>(Point(i%4, floor(i/4))) = vt.at<float>(Point(i, vt.rows-1));
	}
	return camera;
}

/**
* Obtener una matriz normalizada en el espacio proyectivo a partir de la dada como parámetro @matrix
*/
void normalizeMatrix(Mat matrix, Mat &normal){
	float val = matrix.at<float>(Point(matrix.cols-1, matrix.rows-1));
	normal = Mat(matrix.rows, matrix.cols, CV_32FC1);

	for(int i=0; i<matrix.rows; i++){
		for(int j=0; j<matrix.cols; j++){
			normal.at<float>(Point(j,i)) = (matrix.at<float>(Point(j,i)))/val;
		}
	}
}

/**
* Estima el error usando la norma de Frobenius dada la cámara aleatoría y la cámara simulada.
*/
float estimatedError(Mat p1, Mat p2){
	Mat normP1, normP2;

	normalizeMatrix(p1, normP1);
	normalizeMatrix(p2, normP2);

	float inc = 0;

	for(int i=0; i<3; i++)
		for(int j=0; j<4; j++)
				inc += pow(normP1.at<float>(Point(j,i)) - normP2.at<float>(Point(j,i)), 2);

	return sqrt(inc);
}

/**
* Primero se crea la cámara aleatoria, saca los puntos 3D, los proyecta y obtiene que puntos se corresponden en la imagen. Repite el paso anterior pero con la cámara simulada. Estima el error entre ambas  y finalmente genera la imagen correspondiente a los puntos de ambas cámaras.
*/
Mat drawProjectionPoints(){
	srand(time(NULL));

	Mat randomCamera = generateRandomCamera();
	vector<Point3f> pattern = pointPattern();
	Mat aux = Mat::zeros(512, 512, CV_32FC3);
	Size size = Size(512,512);

	vector<Point2f> projections = projectPoints(randomCamera, pattern);
	
	Mat camera = obtainMatrixCamera(projections, pattern);
	vector<Point2f> cameraProjections = projectPoints(camera, pattern);
	
	vector<Point2f> randomFinalPoints = resizePoints(projections, size);
	vector<Point2f> simulateFinalPoints = resizePoints(cameraProjections, size);

	float err = estimatedError(randomCamera, camera);
	cout << "El error entre la cámara aleatoria y estimada es de " << err << endl;

	Scalar color = Scalar(0,0,255);
	for (unsigned int i = 0; i < randomFinalPoints.size(); i++) 
		circle(aux, randomFinalPoints[i], 4, color, -1, CV_AA);

	color = Scalar(0,255,0);
	for (unsigned int i = 0; i < simulateFinalPoints.size(); i++) 
		circle(aux, simulateFinalPoints[i], 2, color, -1, CV_AA);
	
	return aux;
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////Ejercicio 2/////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/**
* Dadas las imágenes @images y mediante la función findChessboardCorners(), busca aquellas que son válidas para calibrar una cámara. El tamaño del tablero que se ha seleccionado es de 13x12. Va guardando los puntos 2D que se van obteniendo de la función mencionada anteriormente y también va generando los puntos 3D mediante el patrón elegido.
*/
vector<Mat> obtainCalibratedPoints(vector<Mat> images, vector<vector<Point2f> > &points, vector<vector<Point3f> > &validWorldPoints){
	vector<Mat> validImages;
	vector<Point2f> corners;
	Size size = Size(13,12);
	Size winSize = Size(5, 5), zeroZone = Size(-1, -1);
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1);
	
	for (unsigned i = 0; i < images.size(); i++) {
		Mat img_gray, aux;
		cvtColor(images[i], img_gray, COLOR_BGR2GRAY);
		if(findChessboardCorners(img_gray, size, corners)){
			img_gray.convertTo(img_gray, CV_32F);
			cornerSubPix(img_gray, corners, winSize, zeroZone, criteria);
			points.push_back(corners);
			images[i].copyTo(aux);
			drawChessboardCorners(aux, size, Mat(corners), 1);
			validImages.push_back(aux);
		
			vector<Point3f> worldPoints;
			for (unsigned i = 0; i < corners.size(); i++) {
				Point3f point = Point3f(i%size.width*100, i/size.width*100, 0);
				worldPoints.push_back(point);
			}
			validWorldPoints.push_back(worldPoints);
		}
	}
	return validImages;
}

/**
* Con las imágenes válidas para calibrar la cámara se pretende obtener el error con/sin distorsión y sus diferentes parámetros así como la matriz cámara y el vector de rotaciones y traslaciones.
*/
double myCalibrateCamera(vector<Mat> images, Mat &camera, vector<Mat> &rotVec, vector<Mat> &trasVec, int type = NO_DIST){
	vector<vector<Point2f> > points;
	vector<vector<Point3f> > worldPoints;
	vector<Mat> validImages = obtainCalibratedPoints(images, points, worldPoints);
	Size size = Size(validImages[0].cols, validImages[0].rows);
	camera = Mat(3, 3, CV_32F);
	Mat coefs = Mat(8, 1, CV_32F);
	double error;

	switch(type){
		case NO_DIST:
			error = calibrateCamera(worldPoints, points, size, camera, coefs, rotVec, trasVec, CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_FIX_K1 | CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6);
		break;
		case RAD_DIST:
			error = calibrateCamera(worldPoints, points, size, camera, coefs, rotVec, trasVec, CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_RATIONAL_MODEL);
		break;
		case TANG_DIST:
			error = calibrateCamera(worldPoints, points, size, camera, coefs, rotVec, trasVec, CV_CALIB_FIX_K1 | CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6);
		break;
		case TANG_AND_RAD_DIST:
			error = calibrateCamera(worldPoints, points, size, camera, coefs, rotVec, trasVec, CV_CALIB_RATIONAL_MODEL);
		break;
	}

 	return error;
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////Ejercicio 3/////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/**
* Obtención de los puntos válidos, es decir, aquellos en los que la máscara o estado @mask sea 1 además de que el tamaño de este nuevo vector de puntos ha de ser menor de 200.
*/
vector<Point2f> fundamentalMatPoints(vector<Point2f> points, vector<uchar> mask){
	vector<Point2f> p;
	int cnt=0;
	for (unsigned i = 0; i < points.size(); i++){
		if(mask[i] == 1 && cnt<200){
			p.push_back(points[i]);
			cnt++;
		}
	}
	return p;
}
/**
* Error obtenido como la media de la didstancia ortogonal entre los puntos soporte y sus líneas epipolares.
*/
float epipolarError(vector<Point2f> points, vector<Point3f> lines){
	float inc = 0;
	for(unsigned i = 0; i < points.size(); i++){
		float dist = (fabs(lines[i].x*points[i].x+lines[i].y*points[i].y+lines[i].z))/sqrt(lines[i].x*lines[i].x+lines[i].y*lines[i].y);
		inc+=dist;
	}
	return inc / points.size();
	
}

/**
* Dibuja en las imágenes correspondientes las líneas epipolares de cada imagen. Para ello Se obtiene la matriz F mediante la función de OpenCV findFundamentalMat() con 8 puntos y RANSAC.
*/
vector<Mat> epipolarPoints(Mat img1, Mat img2, vector<Point2f> points1, vector<Point2f> points2){
	vector<Mat> epiImages;
	vector<Point2f> fPoints1, fPoints2;
	vector<Point3f> lines1, lines2;
	vector<uchar> mask;
	RNG rng(12345);
	Mat image1, image2;
	img1.copyTo(image1);
	img2.copyTo(image2);
	
	Mat F = findFundamentalMat(points1, points2, CV_FM_8POINT | CV_FM_RANSAC, 0.3, 0.95, mask);
	fPoints1 = fundamentalMatPoints(points1, mask);
	fPoints2 = fundamentalMatPoints(points2, mask);

	computeCorrespondEpilines(fPoints1, 1, F, lines2);
	computeCorrespondEpilines(fPoints2, 2, F, lines1);

	for(unsigned i = 0; i < lines1.size(); i++){
		Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		Point2f p1, p2;
		p1 = Point2f(image1.cols, (-lines1[i].z-lines1[i].x*image1.cols)/lines1[i].y);
		p2 = Point2f(0, (-lines1[i].z)/lines1[i].y);
		line(image1, p1, p2, color); 
		circle(image1, fPoints1[i], 3, color, -1, CV_AA);
	}

	for(unsigned i = 0; i < lines2.size(); i++){
		Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		Point2f p1, p2; 
		p1 = Point2f(image2.cols, (-lines2[i].z-lines2[i].x*image2.cols)/lines2[i].y);
		p2 = Point2f(0, (-lines2[i].z)/lines2[i].y);
		line(image2, p1, p2, color); 
		circle(image2, fPoints2[i], 3, color, -1, CV_AA);
	}
	epiImages.push_back(image1);
	epiImages.push_back(image2);
	epiImages.push_back(F);

	float err1 = epipolarError(fPoints1, lines1);
	float err2 = epipolarError(fPoints2, lines2);

	cout << "El error de la primera imagen es de " << err1 << endl;
	cout << "El error de la segunda imagen es de " << err2 << endl;
	
	return epiImages;
}


/**
* Función ampliada de la práctica 2 *
* Uso del detector AKAZE/KAZE/ORB/BRISK y obtención dekeypoints y descriptores para @img1 y @img2 donde posteriormente
se buscará el match entre ellos según el número de correspondencias @correspondences y/o las líneas epipolares. Basicamente
genera la secuencia de correspondencias y el panorama de juntar éstas en las dos imágenes.
*/
vector<Mat> myDetector(Mat img1, Mat img2, unsigned int correspondences, vector<Point2f> &points1, vector<Point2f> &points2, int mode = ALL, int type = TYPE_BRISK){
	vector<Mat> result;
  	Mat img1_gray, img2_gray, descriptors1, descriptors2, correspondence, mosaic;
    vector<KeyPoint> keyP1, keyP2, good_keyP1, good_keyP2;
    vector<DMatch> good_matches, matches;
	int index, queryIdx, trainIdx;

  	if(type2str(img1.type()) != "8UC1")
  		cvtColor(img1, img1_gray, COLOR_BGR2GRAY);
  	else img1.copyTo(img1_gray);
  	
  	if(type2str(img2.type()) != "8UC1")
  		cvtColor(img2, img2_gray, COLOR_BGR2GRAY);
  	else img2.copyTo(img2_gray);

	if(type == TYPE_AKAZE){
		Ptr<AKAZE> detector = AKAZE::create();
		detector->detectAndCompute(img1_gray, noArray(), keyP1, descriptors1);
		detector->detectAndCompute(img2_gray, noArray(), keyP2, descriptors2);
		BFMatcher matcher(NORM_HAMMING, true);
		matcher.match(descriptors1, descriptors2, matches);
	}
	else if(type == TYPE_KAZE){
		Ptr<Feature2D> detector = KAZE::create();
		detector->detectAndCompute(img1_gray, noArray(), keyP1, descriptors1);
		detector->detectAndCompute(img2_gray, noArray(), keyP2, descriptors2);
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
		matcher->match(descriptors1, descriptors2, matches);
 	}

 	else if(type == TYPE_ORB){
		Ptr<ORB> detector = ORB::create(600, 1.2f, 2, 100);
		detector->detectAndCompute(img1_gray, noArray(), keyP1, descriptors1);
		detector->detectAndCompute(img2_gray, noArray(), keyP2, descriptors2);
		BFMatcher matcher(NORM_HAMMING, true);
		matcher.match(descriptors1, descriptors2, matches);

	}
	
	else if(type == TYPE_BRISK){
 		Ptr<BRISK> detector = BRISK::create(10, 0);
		detector->detectAndCompute(img1_gray, noArray(), keyP1, descriptors1);
		detector->detectAndCompute(img2_gray, noArray(), keyP2, descriptors2);
		BFMatcher matcher(NORM_HAMMING, true);
		matcher.match(descriptors1, descriptors2, matches);
	}

    Mat distances(matches.size(), 1, CV_32FC1), aux;

	for( unsigned i = 0; i < matches.size(); i++ ) 
		distances.at<float>(i,0) = matches[i].distance;

	sortIdx(distances,aux,CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

	if(correspondences < 0) correspondences = matches.size();

	for(unsigned i = 0; i < matches.size(); i++ ){
		index = aux.at<int>(i,0);
		good_matches.push_back(matches[index]);
		
		queryIdx = matches[index].queryIdx;
		trainIdx = matches[index].trainIdx;	

		good_keyP1.push_back(keyP1[queryIdx]);
		good_keyP2.push_back(keyP2[trainIdx]);

		good_matches[i].queryIdx = i;
		good_matches[i].trainIdx = i;

		points1.push_back(good_keyP1[i].pt);
		points2.push_back(good_keyP2[i].pt);

	}

	if(mode == ALL || mode == ONLY_CORRESPONDENCES){
		drawMatches(img1, good_keyP1, img2, good_keyP2, good_matches, correspondence);
		result.push_back(correspondence);
	}
	
	if(mode == ALL || mode == ONLY_PANORAMA){
		blendImages(img1, img2, mosaic, points1, points2);
		result.push_back(mosaic);
	}
	if(mode == ALL || mode == ONLY_EPIPOLAR || mode == ONLY_FUNDAMENTAL_MAT){
		vector<Mat> epiImages = epipolarPoints(img1, img2, points1, points2);
		if(mode == ONLY_EPIPOLAR){
			Mat auxImg = epiImages[0];
			hconcat(auxImg, epiImages[1], auxImg);
			result.push_back(auxImg);
		}
		else if(mode == ONLY_FUNDAMENTAL_MAT)
			result.push_back(epiImages[2]);

	}

	return result;
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////Ejercicio 4/////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/**
* Obtiene la matriz esencial resultante de dos imágenes @img1 e @img2 así como los vectores de las correspondencias. 
*/
vector<Mat> essentialMat(Mat img1, Mat img2, vector<Point2f> &points1, vector<Point2f> &points2, Mat K){
	vector<Mat> resultMats;
	Mat essential;

	resultMats.push_back(myDetector(img1, img2, -1, points1, points2, ONLY_FUNDAMENTAL_MAT, TYPE_BRISK)[0]);
	
	essential = K.t() * resultMats[0] * K;

	resultMats.push_back(essential);

	return resultMats;
}

/**
* Devuelve la matriz resultante del movimiento traslación de la matriz esencial
*/
Mat obtainTraslationVector(Mat essential){
	Mat enorm, tnorm, eye, tras;
	eye = Mat::eye(3, 3, CV_64F);
	tras = Mat(1, 3, CV_64F);
	int max = 0;

	enorm = essential*essential.t()/(trace(essential*essential.t())[0]/2);
	tnorm = -enorm + eye;

	if(tnorm.at<double>(Point2f(1,1)) > tnorm.at<double>(Point2f(0,0))) max = 1;
	if(tnorm.at<double>(Point2f(2,2)) > tnorm.at<double>(Point2f(max,max))) max = 2;

	tras = (1/sqrt(tnorm.at<double>(Point2f(max,max))))*tnorm.row(max);

	return tras;
}

/**
* Devuelve la matriz resultante del movimiento rotación de la matriz esencial y el vector @v
*/
Mat obtainRotationVec(Mat essential, Mat v){
	Mat rotation, w[3], enorm;
	rotation = Mat(3,3,CV_64F);
	enorm = essential/sqrt((trace(essential*essential.t())[0]/2));

	for(unsigned i = 0; i < 3; i++ ){
		Mat row = enorm.row(i);
		w[i] = row.cross(v);
	}

	for(unsigned i = 0; i < 3; i++ )
		rotation.row(i) = w[i] + w[(i+1)%3].cross(w[(i+2)%3]);
	
	return rotation;
}

/**
* Genera la profundidad negativa dependiendo de los puntos y los movimientos de rotación @R y traslación @t y distancias focales.
*/
int negativeDepth(vector<Point2f> &points1, vector<Point2f> &points2, Mat &t, Mat &R, float focalDistance1, float focalDistance2){
	int cnt = 0;

	for(unsigned i = 0; i < points1.size(); i++ ){
		Mat h = Mat(1, 3, CV_64F);
		h.at<double>(0) = points1[i].x;
		h.at<double>(1) = points1[i].y;
		h.at<double>(2) = 0;

		double ldepth = focalDistance2*(focalDistance1*t.dot(R.row(0)-points2[i].x*R.row(2)))/(focalDistance2*h.dot(R.row(0)-points2[i].x*R.row(2)));

		double rdepth = R.row(2).dot(focalDistance1/ldepth*h-t);

		if(ldepth < 0 || rdepth < 0) cnt++;

	}
	return cnt;
}

/**
* Obtiene los movimientos de Rotación y traslación de la matriz esencial.
*/
void obtainMovements(Mat &essential, vector<Point2f> &points1, vector<Point2f> &points2, Mat &t, Mat &R, float focalDistance1, float focalDistance2){
	Mat positiveTras = obtainTraslationVector(essential);
	Mat negativeTras = -positiveTras;

	Mat fRot = obtainRotationVec(essential, positiveTras);
	Mat sRot = obtainRotationVec(-1*essential, positiveTras);
	Vec4i cntDepth;
 
	for(unsigned i = 0; i < 4; i++ ){
		Mat auxTras, auxRot;
		if(i<2) auxTras = positiveTras;
		else auxTras = negativeTras;
		if(i%2==0) auxRot = fRot;
		else auxRot = sRot;

		cntDepth[i] = negativeDepth(points1, points2, auxTras, auxRot, focalDistance1, focalDistance2);
	}
	int minNegDepthPos;
	double minNegDepth;

	minMaxIdx(cntDepth, &minNegDepth, NULL, &minNegDepthPos);

	switch(minNegDepthPos)
	{
		case 0:
			t = positiveTras; R = fRot;
		break;
		case 1:
			t = positiveTras; R = sRot;
		break;
		case 2:
			t = negativeTras; R = fRot;
		break;
		case 3:
			t = negativeTras; R = sRot;
		break;
	}
}

	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////Ejercicio 5/////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////

int main() {

	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////Ejercicio 1/////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
	
	cout << endl << "//////////////////////Ejercicio 1//////////////////////" << endl;
	vector<Mat> images;

	cout << "Obteniendo las cámaras y generando la imagen resultante..." << endl;

	images.push_back(drawProjectionPoints());

	showImages(images);
	images.clear();


	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////Ejercicio 2/////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
	
	cout << endl << "//////////////////////Ejercicio 2//////////////////////" << endl;

	vector<Mat> chess, validChess, auxRot, auxTras, cameras, res;
	vector<vector<Mat> > rotVec, trasVec;
	Mat auxCamera;
	vector<vector<Point2f> > points;
	vector<vector<Point3f> > worldPoints;

	for (unsigned i = 1; i <= 25; i++) {
			stringstream number;
			number << i; 
			string path = "T3/imagenes/Image" + number.str();
			string extension = ".tif";
			string filename = path + extension;
			chess.push_back(imread(filename));
	}

	cout << "Obteniendo las imágenes válidas para calibrar una cámara..." << endl;
	validChess = obtainCalibratedPoints(chess, points, worldPoints);
	Mat h1 = validChess[0];
	Mat h2 = validChess[2];
	Mat image;
	hconcat(h1, validChess[1], h1);
	hconcat(h2, validChess[3], h2);
	vconcat(h1, h2, image);
	res.push_back(image);
	cout << "A continuación se mostrarán las imágenes obtenidas." << endl;
	showImages(res);

	double noDistErr, distRadErr, distTangErr, distRadAndTangErr;
	cout << "Generando los parámetros y el error sin distorsión..." << endl;
	noDistErr = myCalibrateCamera(chess, auxCamera, auxRot, auxTras, NO_DIST);
	cameras.push_back(auxCamera); rotVec.push_back(auxRot); trasVec.push_back(auxTras);

	auxCamera.release(); auxRot.clear(); auxTras.clear();
	cout << "El error sin distorsión (1) es de " << noDistErr << endl << endl;
	cout << "Generando los parámetros y el error con distorsión radial..." << endl;
	distRadErr = myCalibrateCamera(chess, auxCamera, auxRot, auxTras, RAD_DIST);
	cameras.push_back(auxCamera); rotVec.push_back(auxRot); trasVec.push_back(auxTras);
	auxCamera.release(); auxRot.clear(); auxTras.clear();
	cout << "El error con distorsión radial (2) es de " << distRadErr << endl << endl;
	
	cout << "Generando los parámetros y el error con distorsión tangencial..." << endl;
	distTangErr = myCalibrateCamera(chess, auxCamera, auxRot, auxTras, TANG_DIST);
	cameras.push_back(auxCamera); rotVec.push_back(auxRot); trasVec.push_back(auxTras);
	auxCamera.release(); auxRot.clear(); auxTras.clear();
	cout << "El error con distorsión tangencial (3) es de " << distTangErr << endl << endl;
	
	cout << "Generando los parámetros y el error con distorsión tangencial y radial..." << endl;
	distRadAndTangErr = myCalibrateCamera(chess, auxCamera, auxRot, auxTras, TANG_AND_RAD_DIST);
	cameras.push_back(auxCamera); rotVec.push_back(auxRot); trasVec.push_back(auxTras);
	cout << "El error con distorsión tangencial y radial (4) es de " << distRadAndTangErr << endl << endl;
	

	/*cout << endl << "Los valores intrínsecos de la cámara son: " << endl << auxCamera << endl;
	cout << endl << "Los valores extrinsecos son: " << endl;
	for(unsigned i = 0; i < rotVec.size(); i++){
		cout << "Rotación para distorsión (" << i+1 << ")" << endl << endl;
		for(unsigned j = 0; j < rotVec[i].size(); j++){
			cout << endl << "Para la imagen " << j << endl;
			cout << rotVec[i][j];
		}
		cout << endl << "************************" <<endl;
		cout << "Traslación para distorsión (" << i+1 << ")" << endl << endl;
		for(unsigned j = 0; j < trasVec[i].size(); j++){
			cout << endl << "Para la imagen " << j << endl;
			cout << trasVec[i][j];
		}
		cout << endl << "************************" <<endl;
	}
	*/

	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////Ejercicio 3/////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////

	cout << endl << "//////////////////////Ejercicio 3//////////////////////" << endl;
	vector<Mat> vmort;
	vector<Point2f> points1, points2;

	Mat vmort1 = imread("T3/imagenes/Vmort1.pgm");
	Mat vmort2 = imread("T3/imagenes/Vmort2.pgm");

	vmort.push_back(vmort1); vmort.push_back(vmort2);

	cout << "Obteniendo los puntos en correspondencia válidos y dibujando las líneas epipolares..." << endl;
	images = myDetector(vmort1, vmort2, -1, points1, points2, ONLY_EPIPOLAR, TYPE_BRISK);
	cout << "Mostrando las imágenes con las líneas epipolares" << endl;
	showImages(images);

	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////Ejercicio 4/////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
 
 	cout << endl << "//////////////////////Ejercicio 4//////////////////////" << endl;
	vector<Mat> essentials, t, R;
	vector<vector<Point2f> > firstPoints, secondPoints;
	Mat K;

	Mat rdimage001 = imread("T3/imagenes/rdimage.000.ppm");
	Mat rdimage002 = imread("T3/imagenes/rdimage.001.ppm");
	Mat rdimage003 = imread("T3/imagenes/rdimage.004.ppm");

	double aux[3][3] = { {1839.6300000000001091, 0, 1024.2000000000000455}, {0, 1848.0699999999999363, 686.5180000000000291}, {0,0,1} };

	K = Mat(3, 3, CV_64FC1, &aux);
	
	cout << "A continuación se va a estimar la matriz esencial de cada pareja de imágenes como los movimientos." <<endl;
	for(unsigned i=0; i<3; i++){
		Mat img1, img2, auxT, auxR;
		vector<Point2f> fPoints, sPoints;
		vector<Mat> results;

		switch(i){
			case 0:
				rdimage001.copyTo(img1);
				rdimage002.copyTo(img2);
			break;
			case 1:
				rdimage001.copyTo(img1);
				rdimage003.copyTo(img2);
			break;
			case 2:
				rdimage002.copyTo(img1);
				rdimage003.copyTo(img2);
			break;
		}

		cout << "***********************************************" <<endl;
		cout << "PAREJA " << i+1 << endl;

		results = essentialMat(img1, img2, fPoints, sPoints, K);
		essentials.push_back(results[1]);
		firstPoints.push_back(fPoints);	secondPoints.push_back(sPoints); 

		double focalDistance = K.at<double>(Point(0,0));
		
		obtainMovements(essentials[i], firstPoints[i], secondPoints[i], auxT, auxR, focalDistance, focalDistance);
		t.push_back(auxT); R.push_back(auxR);
		auxT.release(); auxR.release();
		
		cout << "Para la pareja " << i+1 << " la matriz de rotación es de:" << endl << R[i] << endl; 
		cout << "El vector de traslación:" << endl << t[i] << endl << endl;

	}
	
	return 0;
}
