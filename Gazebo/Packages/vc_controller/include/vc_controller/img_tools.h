#ifndef IMG_TOOLS_H
#define IMG_TOOLS_H


#include <iostream>
#include <string>
#include <vector>
#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>

namespace vcc
{
    //  Estructura de datos que contiene la información 
    //      de la cámara
    typedef struct parameters {

        // Image proessing parameters
        float feature_threshold=0.5;
        int nfeatures=250;
        float scaleFactor=1.2;
        int nlevels=8;
        int edgeThreshold=15; // Changed default (31);
        int firstLevel=0;
        int WTA_K=2;
        cv::ORB::ScoreType scoreType=cv::ORB::HARRIS_SCORE;
        int patchSize=30;
        int fastThreshold=20;
        float flann_ratio=0.7;

        // Camera parameters
        cv::Mat K;

    } parameters;

    //  Estructura de datos referente a los resultados
    //      de los emparejamientos de puntos y la homografía
    typedef struct matching_result{
        cv::Mat H;      //  Homografía
        cv::Mat img_matches;    // Imagen para salida de matches
        cv::Mat p1;     // puntos de imagen (u,v)
        cv::Mat p2;     // puntos de imagen (u.v)
        double mean_feature_error=1e10;
    } matching_result;

    //  Estructura que contiene la información de
    //      la pose referencia ()
    typedef struct desired_configuration {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> kp;
        cv::Mat img;
    } desired_configuration;

    //  Calcula la Homografía entre 
    //      la cofiguración deseada y la imagen ingresada 
    //  INPUT:
    //      img = imagen ingresada
    //      params = parámetros de la cámara
    //      desired_configuration = información de referencia
    //  OUTPUT:
    //      result = resultados
    int compute_homography(
        const cv::Mat & img,
        const parameters & params, 
        const desired_configuration & desired_configuration,
        matching_result& result);
    
    //  Selecciona una pose a partir de poses previamente calculadas
    //  INPUT:
    //      Rs = Rotaciones candidatas
    //      Ts = Traslaciones candidatas
    //      Ns = Normales al plano obsevado candidatas
    //      result = resultados
    //      selected = bandera: indica si se ha seleccionado un candidato 
    //                          previa al cálculo
    //  OUTPUT:
    //      Rbest = mejor rotación 
    //      tbest = mejor traslación
    int select_decomposition(
        const std::vector<cv::Mat> & Rs,
        const std::vector<cv::Mat> &Ts,
        const std::vector<cv::Mat> &Ns,
        const matching_result& result,
        bool & selected,
        cv::Mat & Rbest,
        cv::Mat & tbest);
    
    //  Calcula los emparejamientos entre dos imágenes
    //  INPUT:
    //      img = imagen ingresada
    //      params = parámetros de la cámara
    //      desired_configuration = información de referencia
    //  OUTPUT:
    //      result = resultados
    int compute_descriptors(
        const cv::Mat&img,
        const parameters & params, 
        const desired_configuration & desired_configuration,
        matching_result& result);

    //  Calcula los ángulos de Euler
    //      a partir de una matriz de rotación
    cv::Vec3f rotationMatrixToEulerAngles(const cv::Mat &R);
    
    //  Aplica la normalización de los puntos de imagen in situ
    //      sobre un objeto de sesultado
    //      TODO: hacer un metodo propio de vc_matching_result
    void camera_norm(
        const parameters & params, 
        matching_result& result);
    
    //  Calcula la matriz de interacción para los puntos actuales
    //      de un objeto de resultado.
    //      TODO: hacer un método propio de vc_matching_result
    //      TODO: Z estimada en forma de vector.
    cv::Mat interaction_Mat(cv::Mat & p, cv::Mat & Z);

    //  Calcula la Pseudo Inversa Moore Penrose de L
    //      Calcula el determinante en el proceso
    //  Precaución:
    //      Si det < 10^-9 entonces regresa solo L^T
    cv::Mat Moore_Penrose_PInv(cv::Mat L,double & det);
}
#endif
