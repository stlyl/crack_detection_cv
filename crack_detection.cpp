#include "crack_detection.h"
#pragma execution_character_set("utf-8")

crack_detection::crack_detection(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    QFont font("Arial", 12, QFont::Bold);
    ui.label->setFont(font);
    QFont font2("Arial", 12, QFont::Bold);
    ui.label_2->setFont(font2);
}

crack_detection::~crack_detection()
{}
void crack_detection::PreProcess(const Mat& image, Mat& image_blob)
{
    Mat input;
    image.copyTo(input);
    //数据处理 标准化
    std::vector<Mat> channels, channel_p;
    split(input, channels);
    Mat R, G, B;
    B = channels.at(0);
    G = channels.at(1);
    R = channels.at(2);
    B = (B / 255. - 0.406) / 0.225;
    G = (G / 255. - 0.456) / 0.224;
    R = (R / 255. - 0.485) / 0.229;
    channel_p.push_back(R);
    channel_p.push_back(G);
    channel_p.push_back(B);
    Mat outt;
    merge(channel_p, outt);
    image_blob = outt;
}
void crack_detection::on_actionopen_file_triggered()
{
    QString currentpath = QDir::currentPath();
    QString dlgTitle = "Please Select A File!";
    QString strfilter = "crack Files(*.jpg);;All Files(*.*)";
    QString allfiles = QFileDialog::getOpenFileName(this, dlgTitle, currentpath, strfilter);

    if (allfiles.isEmpty())
    {
        QMessageBox::critical(this, "ERROR", "Failed to Open File", QMessageBox::Yes);
        return;
    }
    QLabel* label = new QLabel();
    QPixmap pixmap(allfiles);;
    QPixmap scaledPixmap = pixmap.scaled(QSize(150, 150), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    label->setFixedSize(150, 150);
    label->setPixmap(scaledPixmap);
    QWidget* widget = new QWidget(ui.scrollArea);
    QVBoxLayout* layout = new QVBoxLayout(widget);
    layout->addWidget(label);
    ui.scrollArea->setWidgetResizable(true);
    ui.scrollArea->setWidget(widget);
}
void crack_detection::on_actionopen_folder_triggered()
{
    QString path = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "F:/Learning_materials/learn_cpp/project/crack_detection",
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    QDir dir(path);
    QStringList files = dir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    int count = 0;

    // 创建一个新的 QWidget 作为容器
    QWidget* containerWidget = new QWidget();
    QVBoxLayout* vLayout = new QVBoxLayout(containerWidget);
    containerWidget->setLayout(vLayout);
    QHBoxLayout* hLayout = new QHBoxLayout();
    hLayout->setAlignment(Qt::AlignCenter);
    vLayout->addLayout(hLayout);
    // 遍历所有文件，只处理后缀为 jpg 的文件
    foreach(QString file, files) {
        if (file.endsWith(".jpg", Qt::CaseInsensitive)) {
            QString filePath = path + "/" + file;
            imagePaths.append(filePath); // 将路径添加到列表
            ++count;
            QLabel* label = new QLabel();
            QPixmap pixmap(filePath);
            QPixmap scaledPixmap = pixmap.scaled(QSize(150, 150), 
                Qt::KeepAspectRatio, Qt::SmoothTransformation);
            label->setFixedSize(150, 150);
            label->setPixmap(scaledPixmap);
            hLayout->addWidget(label);
            if (count % maxImagesPerRow == 0) {
                hLayout = new QHBoxLayout();
                hLayout->setAlignment(Qt::AlignCenter);
                vLayout->addLayout(hLayout);
            }
        }
    }
    ui.scrollArea->setFixedSize(500, 170);
    ui.scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui.scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ui.scrollArea->setWidget(containerWidget);
    ui.textEdit->append("图片数量为：" + QString::number(count));
}
void crack_detection::on_actionstart_testing_triggered()
{
    QWidget* containerWidget2 = new QWidget();
    QVBoxLayout* vLayout2 = new QVBoxLayout(containerWidget2);
    containerWidget2->setLayout(vLayout2);
    QHBoxLayout* hLayout2 = new QHBoxLayout();
    hLayout2->setAlignment(Qt::AlignCenter);
    vLayout2->addLayout(hLayout2);

    std::string onnx_path = "F:/Learning_materials/learn_cpp/project/crack_detection/model/model_softmax_argmax2.onnx";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx_path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    int count = 0;
    foreach(QString path, imagePaths)
    {
        // 1. 读取图像并转换为RGB格式
        //string fileName = "F:/Learning_materials/learn_cpp/project/crack_detection/img/0002.jpg";
        string fileName = path.toStdString();
        Mat bgrImage = cv::imread(fileName, cv::IMREAD_COLOR);
        bgrImage.convertTo(bgrImage, CV_32FC3);
        crack_detection::PreProcess(bgrImage, bgrImage);
        cv::Mat rgbImage;
        cv::cvtColor(bgrImage, rgbImage, cv::COLOR_BGR2RGB);
        // 2. 构建输入batch
        cv::Mat inputBlob = cv::dnn::blobFromImage(bgrImage, 1.0,
            cv::Size(256, 256), cv::Scalar(), false, true);//cv::Scalar(0.485, 0.456, 0.406)
        count++;
        //qDebug() << "Input Blob Size: " << inputBlob.size[0] << " x " << inputBlob.size[1];
        // 3. 加载模型
        // 4. 进行图像推理
        //double t = static_cast<double>(cv::getTickCount());
        net.setInput(inputBlob);
        cv::Mat outputBlob = net.forward();
        //t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
        //QString timeString = QString::number(t);
        //ui.textEdit->append("Inference Time: " + timeString);
        //qDebug() << "output Blob Size: " << outputBlob.size[0] << " x " << outputBlob.size[1] << " x " << outputBlob.size[2];
        outputBlob = outputBlob.reshape(1, { 256, 256 });


        vector<int> flattened;
        for (int i = 0; i < 256; i++)
        {
            for (int j = 0; j < 256; j++)
            {
                // 访问 outputBlob 中第 i 行、第 j 列的元素，并将其转换为整型后添加到 flattened 向量中
                flattened.push_back(static_cast<int>(outputBlob.at<float>(i, j)));
            }
        }
        // 使用 set 来去除重复元素并排序
        std::set<int> uniqueSet(flattened.begin(), flattened.end());

        // 将 set 转换回 vector
        std::vector<int> uniqueArr(uniqueSet.begin(), uniqueSet.end());

        // 使用 qDebug 输出结果
        qDebug() << "Unique elements:";
        for (const auto& element : uniqueArr)
        {
            qDebug() << element;
        }

        qDebug() << "output Blob Size: " << outputBlob.size[0] << " x " << outputBlob.size[1];
        // 5. 解析输出结果
        std::vector<std::string> CLASSES = { "ignore", "crack", "spall", "rebar" };
        std::vector<cv::Vec3b> PALETTE = { cv::Vec3b(0, 0, 0), cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(0, 0, 255) }; // bgr
        cv::Mat color_seg(outputBlob.rows, outputBlob.cols, CV_8UC3, cv::Scalar(0, 0, 0));

        for (int row = 0; row < outputBlob.rows; row++) 
        {
            for (int col = 0; col < outputBlob.cols; col++) 
            {
                float label = outputBlob.at<float>(row, col);
                //qDebug() << QString::number(label);
                //int label = outputBlob.at<int>(row, col);
                assert(label >= 0 && label < PALETTE.size());
                cv::Vec3b color = PALETTE[label];
                color_seg.at<cv::Vec3b>(row, col) = color;
            }
        }
        QImage qImg(color_seg.data, color_seg.cols, color_seg.rows,
            static_cast<int>(color_seg.step), QImage::Format_RGB888);
        QPixmap qPixmap = QPixmap::fromImage(qImg.rgbSwapped());
        QLabel* label = new QLabel();
        QPixmap scaledPixmap = qPixmap.scaled(QSize(150, 150),
            Qt::KeepAspectRatio, Qt::SmoothTransformation);
        label->setFixedSize(150, 150);
        label->setPixmap(scaledPixmap);
        hLayout2->addWidget(label);
        if (count % maxImagesPerRow == 0) {
            hLayout2 = new QHBoxLayout();
            hLayout2->setAlignment(Qt::AlignCenter);
            vLayout2->addLayout(hLayout2);
        }
    }
    ui.scrollArea_2->setFixedSize(500, 170);
    ui.scrollArea_2->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui.scrollArea_2->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ui.scrollArea_2->setWidget(containerWidget2);
    ui.textEdit->append("Inference Successfully!");
 }



    /*
    void crack_detection::on_load_weight_clicked()
{
    QString current_model = ui.comboBox->currentText();
    qDebug() << current_model;
    ui.textEdit->append(current_model + " model and weight loaded successfully!");
}
    std::vector<int> flattened;
    for (int i = 0; i < 256; i++) 
    {
        for (int j = 0; j < 256; j++) 
        {
            // 访问 outputBlob 中第 i 行、第 j 列的元素，并将其转换为整型后添加到 flattened 向量中
            flattened.push_back(static_cast<int>(outputBlob.at<float>(i, j)));
        }
    }
    // 使用 set 来去除重复元素并排序
    std::set<int> uniqueSet(flattened.begin(), flattened.end());

    // 将 set 转换回 vector
    std::vector<int> uniqueArr(uniqueSet.begin(), uniqueSet.end());

    // 使用 qDebug 输出结果
    qDebug() << "Unique elements:";
    for (const auto& element : uniqueArr) 
    {
        qDebug() << element;
    }

*/