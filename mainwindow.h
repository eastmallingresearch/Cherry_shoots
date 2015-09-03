#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_training_clicked();

    void on_pushButton_test_clicked();

    void on_pushButton_close_clicked();


    void read_num_class_data( const char* filename, int var_count, Mat& data, Mat& response );

    void on_pushButton_clicked();

private:
    vector <string> listFile(char folder_name[]);

private:
    Ui::MainWindow *ui;

};


#endif // MAINWINDOW_H

