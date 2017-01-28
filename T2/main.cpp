/**
* Cristóbal Antonio Olivencia Carrión <cristobalolivencia@correo.ugr.es>
*
* VC - Visión por Computador
*
* Trabajo 2
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <cmath>
#include <time.h>

using namespace std;
using namespace cv;

/**
* Enumerados que se han usado para hacer más legible el código
*/
enum {ZEROS = 0, REFLECTION = 1};
enum {ONLY_CORRESPONDENCES = 1, ONLY_PANORAMA = 2, BOTH = 3};
enum {TYPE_AKAZE = 0, TYPE_KAZE = 1};

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

/**
* Uso del detector AKAZE/KAZE y obtener los keypoints y descriptores para @img1 y @img2 donde posteriormente
se buscará el match entre ellos según el número de correspondencias @correspondences. Basicamente
genera la secuencia de correspondencias y el panorama de juntar éstas en las dos imágenes.
*/
vector<Mat> myDetector(Mat img1, Mat img2, int correspondences, int mode = BOTH, int type = TYPE_AKAZE ){
	vector<Mat> result;
  	Mat img1_gray, img2_gray, descriptors1, descriptors2, correspondence, mosaic;
    vector<KeyPoint> keyP1, keyP2, good_keyP1, good_keyP2;
    vector<Point2f> points1, points2;
    vector<DMatch> good_matches, matches;
	int index, queryIdx, trainIdx;

  	cvtColor(img1, img1_gray, COLOR_BGR2GRAY);
  	cvtColor(img2, img2_gray, COLOR_BGR2GRAY);

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

    Mat distances(matches.size(), 1, CV_32FC1), aux;

	for( unsigned i = 0; i < matches.size(); i++ ) 
		distances.at<float>(i,0) = matches[i].distance;

	sortIdx(distances,aux,CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

	for( int i = 0; i < correspondences; i++ ){
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
	if(mode == BOTH || mode == ONLY_CORRESPONDENCES){
		drawMatches(img1, good_keyP1, img2, good_keyP2, good_matches, correspondence);
		result.push_back(correspondence);
	}
	
	if(mode == BOTH || mode == ONLY_PANORAMA){
		blendImages(img1, img2, mosaic, points1, points2);
		result.push_back(mosaic);
	}
	return result;
}

/**
* Dado el vector de imágenes @images que contiene las diferentes imagenes de un mosaico y según
el número de correspondencias @correspondences obtener la imagen resultante que es un panorama de 
dichas imágenes.
*/
Mat getPanorama(vector<Mat> images, int correspondences, int type = TYPE_AKAZE){
	int n_images = images.size();
	Mat aux = images[0], panorama;
	for(int i = 0; i < n_images - 1; i++){
		panorama = myDetector(aux, images[i+1], correspondences, ONLY_PANORAMA, type)[0];
		aux = panorama;
	}
	return panorama;
}

int main() {

	cout << "Cargando imágenes..." << endl;

	Mat Yosemite1 = imread("T2/imagenes/Yosemite1.jpg");
	Mat Yosemite2 = imread("T2/imagenes/Yosemite2.jpg");

	Mat yosemite1 = imread("T2/imagenes/yosemite1.jpg");
	Mat yosemite2 = imread("T2/imagenes/yosemite2.jpg");
	Mat yosemite3 = imread("T2/imagenes/yosemite3.jpg");
	Mat yosemite4 = imread("T2/imagenes/yosemite4.jpg");
	Mat yosemite5 = imread("T2/imagenes/yosemite5.jpg");
	Mat yosemite6 = imread("T2/imagenes/yosemite6.jpg");
	Mat yosemite7 = imread("T2/imagenes/yosemite7.jpg");

	Mat mosaico002 = imread("T2/imagenes/mosaico002.jpg");
	Mat mosaico003 = imread("T2/imagenes/mosaico003.jpg");
	Mat mosaico004 = imread("T2/imagenes/mosaico004.jpg");
	Mat mosaico005 = imread("T2/imagenes/mosaico005.jpg");
	Mat mosaico006 = imread("T2/imagenes/mosaico006.jpg");
	Mat mosaico007 = imread("T2/imagenes/mosaico007.jpg");
	Mat mosaico008 = imread("T2/imagenes/mosaico008.jpg");
	Mat mosaico009 = imread("T2/imagenes/mosaico009.jpg");
	Mat mosaico010 = imread("T2/imagenes/mosaico010.jpg");
	Mat mosaico011 = imread("T2/imagenes/mosaico011.jpg");

	cout << "Imágenes cargadas en memoria. Para cambiar entre imágenes pulse cualquier letra." << endl;

	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////Ejercicio 1/////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
	
	vector<Mat> images, yosemite, mosaico, tablero;
	clock_t begin_time;

	cout << "//////////////////////Ejercicio 1//////////////////////" << endl;

	cout << "Comenzando la obtención de los puntos relevantes de la imagen..." << endl;
	cout << "Pulse cualquier tecla para ver los puntos obtenidos en las diferentes escalas." << endl;

	scalePyramid(Yosemite1,2);
	scalePyramid(Yosemite2,2);

	cout << "Imagen obtenido con círculos alrededor de los puntos y la línea con la dirección del gradiente." << endl;

	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////Ejercicio 2/////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////

	cout << "//////////////////////Ejercicio 2//////////////////////" << endl;

	int correspondences = 30;

	cout << "Generando las " << correspondences << " correspondencias entre las imágenes de Yosemite 1 y 2 con el detector AKAZE..." << endl;
	
	begin_time = clock();
	images.push_back(myDetector(yosemite1, yosemite2, correspondences, ONLY_CORRESPONDENCES)[0]);
	cout << "Tiempo empleado " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << " segundos." << endl;
	
	cout << "Generando las " << correspondences << " correspondencias entre las imágenes de Yosemite 2 y 3 con el detector KAZE..." << endl;
	
	begin_time = clock();
	images.push_back(myDetector(yosemite2, yosemite3, correspondences, ONLY_CORRESPONDENCES, TYPE_KAZE)[0]);
	cout << "Tiempo empleado " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << " segundos." << endl;
	
	cout << "Obtención de las imágenes finalizada." << endl;
	
	showImages(images);
	remove(images);

	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////Ejercicio 3/////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////

	cout << "//////////////////////Ejercicio 3//////////////////////" << endl;
	
	yosemite.push_back(yosemite3); yosemite.push_back(yosemite2);
	yosemite.push_back(yosemite4); yosemite.push_back(yosemite1);

	mosaico.push_back(mosaico005); mosaico.push_back(mosaico004);
	mosaico.push_back(mosaico006); mosaico.push_back(mosaico007);
	mosaico.push_back(mosaico008); mosaico.push_back(mosaico003);
	mosaico.push_back(mosaico002); mosaico.push_back(mosaico009);
	mosaico.push_back(mosaico010); mosaico.push_back(mosaico011);
	
	cout << "Generando el panorama de las " << yosemite.size() << " imágenes de Yosemite con el detector AKAZE..." << endl;

	begin_time = clock();
	images.push_back(getPanorama(yosemite,100));
	cout << "Tiempo empleado " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << " segundos." << endl;
	
	cout << "Panorama de las imágenes Yosemite obtenido." << endl;

	showImages(images);
	remove(yosemite);
	remove(images);

	
	cout << "Generando el panorama de las " << mosaico.size() << " imágenes del paisaje en mosaico con el detector KAZE..." << endl;
	
	begin_time = clock();
	images.push_back(getPanorama(mosaico,50, TYPE_KAZE));
	cout << "Tiempo empleado " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << " segundos." << endl;

	cout << "Panorama de las imágenes mosaico obtenido." << endl;

	showImages(images);
	remove(mosaico);
	remove(images);

	return 0;
}
