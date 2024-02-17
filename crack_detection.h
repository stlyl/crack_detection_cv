#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_crack_detection.h"
#include<QMessageBox>
#include<QFileDialog>
#include <QDebug>
#include <QString>
#include <iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <set>
#include <vector>
#include <QScrollArea>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
using namespace cv;
using namespace std;


class crack_detection : public QMainWindow
{
    Q_OBJECT

public:
    crack_detection(QWidget *parent = nullptr);
    ~crack_detection();

private:
    Ui::crack_detectionClass ui;
    QList<QString> imagePaths; // 图片路径列表
    const int maxImagesPerRow = 3;// 计算每行最多显示的图片数量
    void PreProcess(const Mat& image, Mat& image_blob);
private slots:
    void on_actionopen_file_triggered();
    void on_actionopen_folder_triggered();
    void on_actionstart_testing_triggered();
    
};
