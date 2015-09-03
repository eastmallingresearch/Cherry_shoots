#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <QThread>
#include <algorithm>
#include <iterator>
#include <QFileDialog>

using namespace cv;
using namespace std;

Mat image, image_hsv;
ofstream myfile;
int drag_left, drag_right;
Point point1, point2;
int select_flag;
Rect rect;

int left_num, right_num, total_num;

void onMouse( int event, int x, int y, int, void* )
{

    if (event == CV_EVENT_LBUTTONDOWN && !drag_left)
        {
            /* left button clicked. ROI selection begins */
            point1 = Point(x, y);
            drag_left = 1;
        }

        if (event == CV_EVENT_MOUSEMOVE && drag_left)
        {
            /* mouse dragged. ROI being selected */
            Mat img1 = image.clone();
            point2 = Point(x, y);
            if (x > point1.x && y > point1.y)
            {
                rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 2, 3, 0);
                imshow("Display Image", img1);
            }
            //waitKey(0);
            //img1.release();
        }

        if (event == CV_EVENT_LBUTTONUP && drag_left)
        {
            Mat roiImg;
            point2 = Point(x, y);
            if(x > point1.x && y > point1.y)
            {
                rect = Rect(point1.x,point1.y,x-point1.x,y-point1.y);
                drag_left = 0;
                roiImg = image(rect);
                for (int i = 0; i < roiImg.rows; i++)
                {
                 for (int j = 0; j < roiImg.cols; j++)
                 {
                     myfile<<(int)roiImg.at<Vec3b>(i,j)[0]<<" "<<(int)roiImg.at<Vec3b>(i,j)[1]<<" "<<(int)roiImg.at<Vec3b>(i,j)[2]<<" "<<0<<" "<<endl;
                 }
                }
            total_num += roiImg.rows * roiImg.cols;
            }
            roiImg.release();
        }

        if (event == CV_EVENT_LBUTTONUP)
        {
           /* ROI selected */
            select_flag = 1;
            drag_left = 0;
        }

        if (event == CV_EVENT_RBUTTONDOWN && !drag_right)
             {

                 point1 = Point(x, y);
                 drag_right = 1;
             }

             if (event == CV_EVENT_MOUSEMOVE && drag_right)
             {
                 /* mouse dragged. ROI being selected */
                 Mat img1 = image.clone();
                 point2 = Point(x, y);
                 if(x > point1.x && y > point1.y)
                 {
                     rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 2, 3, 0);
                     imshow("Display Image", img1);
                 }
                 //waitKey(0);
                 //img1.release();
             }

             if (event == CV_EVENT_RBUTTONUP && drag_right)
             {
                 Mat roiImg;
                 point2 = Point(x, y);
                 if(x > point1.x && y > point1.y)
                 {
                     rect = Rect(point1.x,point1.y,x-point1.x,y-point1.y);
                     drag_right = 0;
                     roiImg = image(rect);
                     for (int i = 0; i < roiImg.rows; i++)
                     {
                      for (int j = 0; j < roiImg.cols; j++)
                      {
                         myfile<<(int)roiImg.at<Vec3b>(i,j)[0]<<" "<<(int)roiImg.at<Vec3b>(i,j)[1]<<" "<<(int)roiImg.at<Vec3b>(i,j)[2]<<" "<<1<<" "<<endl;
                      }
                     }
                     total_num += roiImg.rows * roiImg.cols;
                 }
                 roiImg.release();

             }

             if (event == CV_EVENT_RBUTTONUP)
             {
                /* ROI selected */
                 select_flag = 1;
                 drag_right = 0;
             }

}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow::on_pushButton_training_clicked()
{
    left_num = 0;
    right_num = 0;
    total_num = 0;

    QString str = QFileDialog::getExistingDirectory();
    QByteArray ba = str.toLocal8Bit();
    char *c_str = ba.data();
    string slash = "/";

    myfile.open("train.txt");
    string ext = ".jpg";
    vector<string> img_name_train;

    //string directory = "training_img/";
    //char folder_name[30] = "training_img";

    img_name_train = listFile(c_str);

    for (int num = 0; num < img_name_train.size(); num++)
    {
        string img_name = c_str + slash + img_name_train[num];

        Mat HSV, img_thresh;
        image = imread(img_name);

        if( !image.data )
          {
            printf( "No image data \n" );
          //  return -1;
            continue;
          }

    cvtColor(image, image_hsv, CV_RGB2HSV_FULL);
   /* vector<Mat> HSV_split, image_split;
    split(image, image_split);

    threshold(image_split[1], img_thresh, disease_thresh, 255, CV_THRESH_BINARY);

    for (int i = 0; i < image.rows; i++)
    {
      for (int j = 0; j < image.cols; j++)
      {
          if (img_thresh.at<uchar>(i,j) == 0)
          {
              result.at<Vec3b>(i,j)[0] = 0;
              result.at<Vec3b>(i,j)[1] = 0;
              result.at<Vec3b>(i,j)[2] = 255;
          }
          else
          {
              result.at<Vec3b>(i,j) = image.at<Vec3b>(i,j);
          }
      }
    }*/

    //imshow("Split", image_split[1]);
    //imshow( "Display Image", result);

     namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
     //imshow("Compare", image);
     imshow("Display Image", image);
     setMouseCallback( "Display Image", onMouse, 0 );

     QString s = QString::number(total_num);
     ui->label_training_num->setText(s);
     waitKey(0);
     destroyWindow("Display Image");

    }
    QString s = QString::number(total_num);
    ui->label_training_num->setText(s);
}

void MainWindow::on_pushButton_test_clicked()
{

    QString str = QFileDialog::getExistingDirectory();
    QByteArray ba = str.toLocal8Bit();
    char *c_str = ba.data();
    string slash = "/";

    Mat training;
    Mat response;
    read_num_class_data("train.txt", 4, training, response);

    cout<<training.rows<<endl;
    cout<<response.rows<<endl;

    ofstream output_file;
    output_file.open("Ratio.txt");

        Mat layers = Mat(3,1,CV_32SC1);
        int sz = training.cols ;

        layers.row(0) = Scalar(sz);
        layers.row(1) = Scalar(16);
        layers.row(2) = Scalar(1);

        CvANN_MLP mlp;
        CvANN_MLP_TrainParams params;
        CvTermCriteria criteria;

        criteria.max_iter = 1000;
        criteria.epsilon  = 0.00001f;
        criteria.type     = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

        params.train_method    = CvANN_MLP_TrainParams::BACKPROP;
        params.bp_dw_scale     = 0.1f;
        params.bp_moment_scale = 0.1f;
        params.term_crit       = criteria;

        mlp.create(layers,CvANN_MLP::SIGMOID_SYM);
        int i = mlp.train(training, response, Mat(),Mat(),params);                              // Train dataset

        FileStorage fs("mlp.xml",  FileStorage::WRITE); // or xml
        mlp.write(*fs, "mlp"); // don't think too much about the deref, it casts to a FileNode
        ui->label_training->setText("Training finish");
        //mlp.load("mlp.xml","mlp");                                                                //Load ANN weights for each layer


    vector<string> img_name;

    string output_directory = "output_img/";
    img_name = listFile(c_str);

    Mat testing(1, 3, CV_32FC1);
    Mat predict (1 , 1, CV_32F );
    int file_num = 0;

    for(int i = 0; i < img_name.size(); i++)                         //size of the img_name
    {
     ui->progressBar->setValue(i*100/img_name.size());
     string file_name = c_str + slash + img_name[i];

     Mat img_test = imread(file_name);
     Mat img_test_clone = img_test.clone();

     Mat img_thresh, img_thresh_copy, img_HSV, img_gray;
     vector<Mat> img_split;
     cvtColor(img_test_clone, img_HSV, CV_RGB2HSV);
     cvtColor(img_test_clone, img_gray, CV_RGB2GRAY);
     split(img_HSV, img_split);
     threshold(img_split[0], img_thresh, 75, 255, CV_THRESH_BINARY);
     img_thresh_copy = img_thresh.clone();

     Mat hole = img_thresh_copy.clone();
     floodFill(hole, Point(0,0), Scalar(255));
     bitwise_not(hole, hole);
     img_thresh_copy = (img_thresh_copy | hole);

     Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
     Mat open_result;
     morphologyEx(img_thresh, open_result, MORPH_CLOSE, element );

     int infected_num = 0;
     int total_pixels = 0;

     if(img_test.data)
     {
         file_num++;
            for (int m = 0; m < img_test.rows; m++)
            {
             for (int n = 0; n < img_test.cols; n++)
             {
                 if (img_thresh_copy.at<uchar>(m, n) == 255)
                 {
                  total_pixels++;
                  testing.at<float>(0, 0) = (float)img_test.at<Vec3b>(m, n)[0];
                  testing.at<float>(0, 1) = (float)img_test.at<Vec3b>(m, n)[1];
                  testing.at<float>(0, 2) = (float)img_test.at<Vec3b>(m, n)[2];

                  mlp.predict(testing,predict);
                  float a = predict.at<float>(0,0);


                  if (a < 0.4)     //0.4
                  {
                      img_test.at<Vec3b>(m, n)[0] = 0;
                      img_test.at<Vec3b>(m, n)[1] = 0;
                      img_test.at<Vec3b>(m, n)[2] = 255;
                      infected_num++;
                  }
                 }
             }

            }
     float ratio = (float)infected_num / total_pixels * 100;
     output_file<<img_name[i]<<"          "<<(ratio)<<endl;

     string output_file_name = output_directory + img_name[i];
     cout<<output_file_name<<endl;
     imwrite(output_file_name, img_test);

     QImage img_qt = QImage((const unsigned char*)(img_test_clone.data), img_test_clone.cols, img_test_clone.rows, QImage::Format_RGB888);
     QImage img_qt_result = QImage((const unsigned char*)(img_test.data), img_test.cols, img_test.rows, QImage::Format_RGB888);
     //ui->label_original->setPixmap(QPixmap::fromImage(img_qt.rgbSwapped()));
     //ui->label_resulting->setPixmap(QPixmap::fromImage((img_qt_result.rgbSwapped())));

    // imshow("Ori", img_thresh_copy);
     imshow("split", img_test);
     waitKey(0);
     QThread::msleep(100);
     }
     else
     {
      continue;
     }
    }
    ui->progressBar->setValue(100);
    output_file<<endl<<endl<<"Number of processed images:       "<<file_num<<endl;
    cout<<"Test finished!";
}

void MainWindow::on_pushButton_close_clicked()
{
    this->close();
}

vector<string> MainWindow::listFile(char folder_name[])
{
    DIR *pDIR;
    struct dirent *entry;
    vector<string> files;
    if( pDIR=opendir(folder_name) ){
            while(entry = readdir(pDIR)){
                    if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
                   // cout << entry->d_name << "\n";

                    files.push_back(entry->d_name);
            }
            closedir(pDIR);
    }
    return files;
}



void MainWindow::read_num_class_data(const char* filename, int var_count, Mat& data, Mat& response)
{

    ifstream in (filename);
    char buf[50];
    //vector<vector<int> > vec_train;
    vector<int> vec_response;

    vector<int> temp;
    int classes;
    int a, b, c;

    int index = 1;

    ifstream aFile (filename);
    std::size_t lines_count =0;
    std::string line;

    while (std::getline(aFile , line, ' '))
    {
        istringstream ss( line );
        int f;
        ss>>f;

            if(index%var_count != 0)
            {
                temp.push_back(f);
            }
            else
            {
                vec_response.push_back(f);
            }

            index++;
    }


    Mat training_temp((int)(temp.size() / (var_count-1)), var_count-1, CV_32FC1);
    data = training_temp.clone();
    Mat response_temp((int)(temp.size() / (var_count-1)), 1, CV_32FC1);
    response = response_temp.clone();


    int next = 0;
    for (int i = 0; i < (int)(temp.size() / (var_count-1)); i++)
    {
        response.at<float>(i,0) = vec_response[i];
     for (int j = 0; j < var_count-1; j++)
     {
          data.at<float>(i,j) = temp.at(next);
          next++;
     }
    }

}

void MainWindow::on_pushButton_clicked()
{
    QString str = QFileDialog::getExistingDirectory();
    QByteArray ba = str.toLocal8Bit();
    char *c_str = ba.data();
    string slash = "/";

    ofstream output_file;
    output_file.open("Height.txt");

    vector<string> img_name = listFile(c_str);
    for (int i = 0; i < img_name.size(); i++)
    {
        string file_name = c_str + slash + img_name[i];

        QString s = QString::fromStdString(img_name[i]);
        ui->label_infect->setText(s);

        Mat img = imread(file_name);
        Mat img_infect(img.rows, img.cols, CV_8UC1, Scalar(0));
        for (int m = 0; m < img.rows; m++)
        {
            for (int n = 0; n < img.cols; n++)
            {
                if (img.at<Vec3b>(m,n)[0] < 10 && img.at<Vec3b>(m,n)[1] <10 && img.at<Vec3b>(m,n)[2] > 225)
                {
                    img_infect.at<uchar>(m,n) = 255;
                }
            }
        }

        Mat img_infect_copy = img_infect.clone();
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(img_infect, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
        {
            if (it->size() < 10  )
                it=contours.erase(it);
            else
                ++it;
        }


        drawContours(img, contours, -1, Scalar(255, 255, 255), 2);

        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );

        for( int i = 0; i < contours.size(); i++ )
          {
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
          }

        double height = 0;
        double top = 10000;
        double bottom = 0;

        for (int i = 0; i < boundRect.size(); i++)
        {
            double area = (boundRect[i].br().y - boundRect[i].tl().y) * (boundRect[i].br().x - boundRect[i].tl().x);
            if ((boundRect[i].tl().y < (int)img.rows*0.15 && area > 300) || area > 10000)
            {
             rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 0), 2);
             if (boundRect[i].br().y > bottom)
             {
                 bottom = boundRect[i].br().y;
             }
             if (boundRect[i].tl().y < top)
             {
                 top = boundRect[i].tl().y;
             }
            }
        }

        if(bottom == 0)              //no rectangle was selected
        {
            for (int i = 0; i < boundRect.size(); i++)
            {
                double area = (boundRect[i].br().y - boundRect[i].tl().y) * (boundRect[i].br().x - boundRect[i].tl().x);
                if (boundRect[i].tl().y < (int)img.rows*0.06 || area > 6000)
                {
                 rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 0), 2);
                 if (boundRect[i].br().y > bottom)
                 {
                     bottom = boundRect[i].br().y;
                 }
                 if (boundRect[i].tl().y < top)
                 {
                     top = boundRect[i].tl().y;
                 }
                }
            }
        }

          height = (bottom - top) / img.rows * 30;

        output_file<<img_name[i]<<"          "<<height<<endl;

        imshow("Result", img_infect_copy);
        imshow("ori", img);
        waitKey(0);
    }
}
