    // 标准C++库
    #include <iostream>     // 输入输出流
    #include <vector>      // 向量容器
    #include <algorithm>   // 算法库
    #include <chrono>      // 时间相关功能
    #include <iomanip>     // 输入输出格式控制
    #include <sstream>
    #include <array>
    #include <cmath>

    // OpenCV库
    #include <opencv2/opencv.hpp>      // OpenCV主要头文件
    #include <opencv2/dnn/dnn.hpp>     // OpenCV深度学习模块

    // 地平线RDK BPU API
    #include "dnn/hb_dnn.h"           // BPU基础功能
    #include "dnn/hb_dnn_ext.h"       // BPU扩展功能
    #include "dnn/plugin/hb_dnn_layer.h"    // BPU层定义
    #include "dnn/plugin/hb_dnn_plugin.h"   // BPU插件
    #include "dnn/hb_sys.h"           // BPU系统功能

    // 错误检查宏定义
    #define RDK_CHECK_SUCCESS(value, errmsg)                        \
        do                                                          \
        {                                                          \
            auto ret_code = value;                                  \
            if (ret_code != 0)                                      \
            {                                                       \
                std::cout << errmsg << ", error code:" << ret_code; \
                return ret_code;                                    \
            }                                                       \
        } while (0);

    // 模型和检测相关的默认参数定义
    // ========================= 集中配置区（建议只改这里） =========================
    #define DEFAULT_MODEL_PATH "/home/sunrise/Inference-for-RDKx5/autoaim/model/yolov5_nv12_0526_modifier.bin"  // 默认模型路径
    #define DEFAULT_SINGLE_IMAGE_PATH "/home/sunrise/dataset/000031.jpg"  // 单图模式输入图像
    #define DEFAULT_OUTPUT_IMAGE_PATH "cpp_result.jpg"                  // 单图模式输出图像
    #define DEFAULT_VIDEO_PATH "/home/sunrise/dataset/demo.avi"         // 视频模式输入路径
    #define DEFAULT_OUTPUT_VIDEO_PATH "cpp_result.mp4"                   // 视频模式输出路径
    #define DEFAULT_CLASSES_NUM CLASS_CLASS_COUNT  // 默认类别数量
    #define DEFAULT_NMS_THRESHOLD 0.45f    // 非极大值抑制阈值
    #define DEFAULT_SCORE_THRESHOLD 0.80f  // 置信度阈值（对齐 infer_onnx.py）
    #define DEFAULT_NMS_TOP_K 10          // NMS保留的最大框数
    #define DEFAULT_FONT_SIZE 1.0f         // 绘制文字大小
    #define DEFAULT_FONT_THICKNESS 1.0f    // 绘制文字粗细
    #define DEFAULT_LINE_SIZE 2.0f         // 绘制线条粗细
    #define OUTPUT_FEATURE_DIM 22
    #define COLOR_CLASS_COUNT 4
    #define CLASS_CLASS_COUNT 9

    // 运行模式选择
    #define DETECT_MODE 1    // 检测模式: 0-单张图片, 1-实时检测
    #define ENABLE_DRAW 1    // 绘图开关: 0-禁用, 1-启用
    #define LOAD_FROM_DDR 1  // 模型加载方式: 0-从文件加载, 1-从内存加载

    static const std::vector<std::string> DEFAULT_CLASS_NAMES = {
        "G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"
    };

    static const std::vector<std::string> DEFAULT_COLOR_NAMES = {
        "blue", "red", "none", "purple"
    };

    // 顺序：small(10,13..33,23), medium(30,61..59,119), large(116,90..373,326)
    static const std::vector<float> DEFAULT_ANCHORS = {
        10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f,
        30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f,
        116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f
    };
    // ============================================================================

    // 特征图尺度定义 (基于输入尺寸的倍数关系)
    #define H_8 (input_h_ / 8)    // 输入高度的1/8
    #define W_8 (input_w_ / 8)    // 输入宽度的1/8
    #define H_16 (input_h_ / 16)  // 输入高度的1/16
    #define W_16 (input_w_ / 16)  // 输入宽度的1/16
    #define H_32 (input_h_ / 32)  // 输入高度的1/32
    #define W_32 (input_w_ / 32)  // 输入宽度的1/32

    // BPU目标检测类
    class BPU_Detect {
    public:
        // 构造函数：初始化检测器的参数
        // @param model_path: 模型文件路径
        // @param classes_num: 检测类别数量
        // @param nms_threshold: NMS阈值
        // @param score_threshold: 置信度阈值
        // @param nms_top_k: NMS保留的最大框数
        BPU_Detect(const std::string& model_path = DEFAULT_MODEL_PATH,
                    int classes_num = DEFAULT_CLASSES_NUM,
                    float nms_threshold = DEFAULT_NMS_THRESHOLD,
                    float score_threshold = DEFAULT_SCORE_THRESHOLD,
                    int nms_top_k = DEFAULT_NMS_TOP_K);
        
        // 析构函数：释放资源
        ~BPU_Detect();

        // 主要功能接口
        bool Init();  // 初始化BPU和模型
        bool Detect(const cv::Mat& input_img, cv::Mat& output_img);  // 执行目标检测
        bool Release();  // 释放所有资源

    private:
        struct DetectionResult {
            cv::Rect box;
            float confidence;
            int class_id;
            int color_id;
            std::array<cv::Point2f, 4> polygon;
        };

        // 内部工具函数
        bool LoadModel();  // 加载模型文件
        bool GetModelInfo();  // 获取模型的输入输出信息
        bool PreProcess(const cv::Mat& input_img);  // 图像预处理（resize和格式转换）
        bool Inference();  // 执行模型推理
        bool PostProcess();  // 后处理（NMS等）
        void DrawResults(cv::Mat& img);  // 在图像上绘制检测结果
        void PrintResults() const;  // 打印检测结果到控制台

        // 特征图处理辅助函数
        // @param output_tensor: 输出tensor
        // @param height, width: 特征图尺寸
        // @param anchors: 对应尺度的anchor boxes
        // @param conf_thres_raw: 原始置信度阈值
        void ProcessFeatureMap(hbDNNTensor& output_tensor, 
                            int height, int width,
                            const std::vector<std::pair<double, double>>& anchors);

        static inline float Sigmoid(float x) {
            return 1.0f / (1.0f + std::exp(-x));
        }

        // 成员变量（按照构造函数初始化顺序排列）
        std::string model_path_;      // 模型文件路径
        int classes_num_;             // 类别数量（兼容旧参数，不用于新后处理类别数）
        float nms_threshold_;         // NMS阈值
        float score_threshold_;       // 置信度阈值
        int nms_top_k_;              // NMS保留的最大框数
        bool is_initialized_;         // 初始化状态标志
        float font_size_;            // 绘制文字大小
        float font_thickness_;       // 绘制文字粗细
        float line_size_;            // 绘制线条粗细
        
        // BPU相关变量
        hbPackedDNNHandle_t packed_dnn_handle_;  // 打包模型句柄
        hbDNNHandle_t dnn_handle_;               // 模型句柄
        const char* model_name_;                 // 模型名称
        
        // 输入输出张量
        hbDNNTensor input_tensor_;               // 输入tensor
        hbDNNTensor* output_tensors_;            // 输出tensor数组
        hbDNNTensorProperties input_properties_; // 输入tensor属性
        
        // 任务相关
        hbDNNTaskHandle_t task_handle_;          // 推理任务句柄
        
        // 模型输入参数
        int input_h_;                            // 输入高度
        int input_w_;                            // 输入宽度
        
        // 检测结果存储
        std::vector<DetectionResult> detections_;
        std::vector<int> nms_indices_;
        
        // 图像处理参数
        float x_scale_;                          // X方向缩放比例
        float y_scale_;                          // Y方向缩放比例
        int x_shift_;                            // X方向偏移量
        int y_shift_;                            // Y方向偏移量
        cv::Mat resized_img_;                    // 缩放后的图像
        
        // YOLOv5 anchors信息
        std::vector<std::pair<double, double>> s_anchors_;  // 小目标anchors
        std::vector<std::pair<double, double>> m_anchors_;  // 中目标anchors
        std::vector<std::pair<double, double>> l_anchors_;  // 大目标anchors
        
        // 输出处理
        int output_order_[3];                    // 输出顺序映射
        std::vector<std::string> class_names_;   // 类别名称列表
        std::vector<std::string> color_names_;   // 颜色名称列表
    };

    // 构造函数实现
    BPU_Detect::BPU_Detect(const std::string& model_path,
                            int classes_num,
                            float nms_threshold,
                            float score_threshold,
                            int nms_top_k)
        : model_path_(model_path),
        classes_num_(classes_num),
        nms_threshold_(nms_threshold),
        score_threshold_(score_threshold),
        nms_top_k_(nms_top_k),
        is_initialized_(false),
        font_size_(DEFAULT_FONT_SIZE),
        font_thickness_(DEFAULT_FONT_THICKNESS),
        line_size_(DEFAULT_LINE_SIZE),
        task_handle_(nullptr) 
        {
        
        // 初始化类别名称
        class_names_ = DEFAULT_CLASS_NAMES;
        color_names_ = DEFAULT_COLOR_NAMES;
        
        // 初始化anchors
        std::vector<float> anchors = DEFAULT_ANCHORS;
        
        // 设置small, medium, large anchors
        for(int i = 0; i < 3; i++) {
            s_anchors_.push_back({anchors[i*2], anchors[i*2+1]});
            m_anchors_.push_back({anchors[i*2+6], anchors[i*2+7]});
            l_anchors_.push_back({anchors[i*2+12], anchors[i*2+13]});
        }
    }

    // 析构函数实现
    BPU_Detect::~BPU_Detect() {
        if(is_initialized_) {
            Release();
        }
    }

    // 初始化函数实现
    bool BPU_Detect::Init() {
        if(is_initialized_) {
            std::cout << "Already initialized!" << std::endl;
            return true;
        }
        
        auto init_start = std::chrono::high_resolution_clock::now();
        
        if(!LoadModel()) {
            std::cout << "Failed to load model!" << std::endl;
            return false;
        }
        
        if(!GetModelInfo()) {
            std::cout << "Failed to get model info!" << std::endl;
            return false;
        }
        
        is_initialized_ = true;
        
        auto init_end = std::chrono::high_resolution_clock::now();
        float init_time = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start).count() / 1000.0f;
        
        std::cout << "\n============ Model Loading Time ============" << std::endl;
        std::cout << "Total init time: " << std::fixed << std::setprecision(2) << init_time << " ms" << std::endl;
        std::cout << "=========================================\n" << std::endl;
        
        return true;
    }

    // 加载模型实现
    bool BPU_Detect::LoadModel() {
        // 记录总加载时间的起点
        auto load_start = std::chrono::high_resolution_clock::now();

    #if LOAD_FROM_DDR
        // 用于记录从文件读取模型数据的时间
        float read_time = 0.0f;
    #endif
        // 用于记录模型初始化的时间
        float init_time = 0.0f;
        
    #if LOAD_FROM_DDR
        // =============== 从文件读取模型到内存 ===============
        auto read_start = std::chrono::high_resolution_clock::now();
        
        // 打开模型文件
        FILE* fp = fopen(model_path_.c_str(), "rb");
        if (!fp) {
            std::cout << "Failed to open model file: " << model_path_ << std::endl;
            return false;
        }
        
        // 获取文件大小:
        fseek(fp, 0, SEEK_END);// 1. 将文件指针移到末尾
        size_t model_size = static_cast<size_t>(ftell(fp));// 2. 获取当前位置(即文件大小)
        fseek(fp, 0, SEEK_SET);// 3. 将文件指针重置到开头
        
        // 为模型数据分配内存
        void* model_data = malloc(model_size);
        if (!model_data) {
            std::cout << "Failed to allocate memory for model data" << std::endl;
            fclose(fp);
            return false;
        }
        
        // 读取模型数据到内存
        size_t read_size = fread(model_data, 1, model_size, fp);
        fclose(fp);
        
        // 计算文件读取时间
        auto read_end = std::chrono::high_resolution_clock::now();
        read_time = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count() / 1000.0f;
        
        // 验证是否完整读取了文件
        if (read_size != model_size) {
            std::cout << "Failed to read model data, expected " << model_size 
                    << " bytes, but got " << read_size << " bytes" << std::endl;
            free(model_data);
            return false;
        }
        
        // =============== 从内存初始化模型 ===============
        auto init_start = std::chrono::high_resolution_clock::now();
        
        // 准备模型数据数组和长度数组
        const void* model_data_array[] = {model_data};
        int32_t model_data_length[] = {static_cast<int32_t>(model_size)};
        
        // 使用BPU API从内存初始化模型
        RDK_CHECK_SUCCESS(
            hbDNNInitializeFromDDR(&packed_dnn_handle_, model_data_array, model_data_length, 1),
            "Initialize model from DDR failed");
        
        // 释放临时分配的内存
        free(model_data);
        
        // 计算模型初始化时间
        auto init_end = std::chrono::high_resolution_clock::now();
        init_time = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start).count() / 1000.0f;
        
    #else
        // =============== 直接从文件初始化模型 ===============
        auto init_start = std::chrono::high_resolution_clock::now();
        
        // 获取模型文件路径
        const char* model_file_name = model_path_.c_str();
        
        // 使用BPU API从文件初始化模型
        RDK_CHECK_SUCCESS(
            hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_file_name, 1),
            "Initialize model from file failed");
        
        // 计算模型初始化时间
        auto init_end = std::chrono::high_resolution_clock::now();
        init_time = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start).count() / 1000.0f;
    #endif

        // =============== 计算并打印总时间统计 ===============
        auto load_end = std::chrono::high_resolution_clock::now();
        float total_load_time = std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count() / 1000.0f;

        // 打印时间统计信息
        std::cout << "\n============ Model Loading Details ============" << std::endl;
    #if LOAD_FROM_DDR
        std::cout << "File reading time: " << std::fixed << std::setprecision(2) << read_time << " ms" << std::endl;
    #endif
        std::cout << "Model init time: " << std::fixed << std::setprecision(2) << init_time << " ms" << std::endl;
        std::cout << "Total loading time: " << std::fixed << std::setprecision(2) << total_load_time << " ms" << std::endl;
        std::cout << "===========================================\n" << std::endl;

        return true;
    }

    // 获取模型信息实现
    bool BPU_Detect::GetModelInfo() {
        // 获取模型名称列表
        const char** model_name_list;
        int model_count = 0;
        RDK_CHECK_SUCCESS(
            hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle_),
            "hbDNNGetModelNameList failed");
        if(model_count > 1) {
            std::cout << "Model count: " << model_count << std::endl;
            std::cout << "Please check the model count!" << std::endl;
            return false;
        }
        model_name_ = model_name_list[0];
        
        // 获取模型句柄
        RDK_CHECK_SUCCESS(
            hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_),
            "hbDNNGetModelHandle failed");
        
        // 获取输入信息
        int32_t input_count = 0;
        RDK_CHECK_SUCCESS(
            hbDNNGetInputCount(&input_count, dnn_handle_),
            "hbDNNGetInputCount failed");
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_properties_, dnn_handle_, 0),
            "hbDNNGetInputTensorProperties failed");

        if(input_count > 1){
            std::cout << "模型输入节点大于1，请检查！" << std::endl;
            return false;
        }
        if(input_properties_.validShape.numDimensions == 4){
            std::cout << "输入tensor类型: HB_DNN_IMG_TYPE_NV12" << std::endl;
        }
        else{
            std::cout << "输入tensor类型不是HB_DNN_IMG_TYPE_NV12，请检查！" << std::endl;
            return false;
        }
        if(input_properties_.tensorType == 1){
            std::cout << "输入tensor数据排布: HB_DNN_LAYOUT_NCHW" << std::endl;
        }
        else{
            std::cout << "输入tensor数据排布不是HB_DNN_LAYOUT_NCHW，请检查！" << std::endl;
            return false;
        }
        // 获取输入尺寸
        input_h_ = input_properties_.validShape.dimensionSize[2];
        input_w_ = input_properties_.validShape.dimensionSize[3];
        if (input_properties_.validShape.numDimensions == 4)
        {
            std::cout << "输入的尺寸为: (" << input_properties_.validShape.dimensionSize[0];
            std::cout << ", " << input_properties_.validShape.dimensionSize[1];
            std::cout << ", " << input_h_;
            std::cout << ", " << input_w_ << ")" << std::endl;
        }
        else
        {
            std::cout << "输入的尺寸不是(1,3,640,640)，请检查！" << std::endl;
            return false;
        }
        
        // 获取输出信息并调整输出顺序
        int32_t output_count = 0;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputCount(&output_count, dnn_handle_),
            "hbDNNGetOutputCount failed");
        
        // 分配输出tensor内存
        output_tensors_ = new hbDNNTensor[output_count];
        memset(output_tensors_, 0, sizeof(hbDNNTensor) * output_count);
        
        // =============== 调整输出头顺序映射 ===============
        // YOLOv5有3个输出头，分别对应3种不同尺度的特征图
        // 需要确保输出顺序为: 小目标(8倍下采样) -> 中目标(16倍下采样) -> 大目标(32倍下采样)
        
        // 初始化默认顺序
        output_order_[0] = 0;  // 默认第1个输出
        output_order_[1] = 1;  // 默认第2个输出
        output_order_[2] = 2;  // 默认第3个输出

        // 定义期望的输出特征图尺寸和通道数
        int32_t expected_shapes[3][3] = {
            {H_32, W_32, 3 * OUTPUT_FEATURE_DIM},   // 20x20
            {H_16, W_16, 3 * OUTPUT_FEATURE_DIM},   // 40x40
            {H_8,  W_8,  3 * OUTPUT_FEATURE_DIM}    // 80x80
        };

        // 遍历每个期望的输出尺度
        for(int i = 0; i < 3; i++) {
            // 遍历实际的输出节点
            for(int j = 0; j < 3; j++) {
                // 获取当前输出节点的属性
                hbDNNTensorProperties output_properties;
                RDK_CHECK_SUCCESS(
                    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, j),
                    "Get output tensor properties failed");
                
                // 获取实际的特征图尺寸和通道数
                int32_t actual_h = output_properties.validShape.dimensionSize[1];
                int32_t actual_w = output_properties.validShape.dimensionSize[2];
                int32_t actual_c = output_properties.validShape.dimensionSize[3];

                // 如果实际尺寸和通道数与期望的匹配
                if(actual_h == expected_shapes[i][0] && 
                actual_w == expected_shapes[i][1] && 
                actual_c == expected_shapes[i][2]) {
                    // 记录正确的输出顺序
                    output_order_[i] = j;
                    break;
                }
            }
        }

        // 打印输出顺序映射信息
        std::cout << "\n============ Output Order Mapping ============" << std::endl;
        std::cout << "20x20 head: output[" << output_order_[0] << "]" << std::endl;
        std::cout << "40x40 head: output[" << output_order_[1] << "]" << std::endl;
        std::cout << "80x80 head: output[" << output_order_[2] << "]" << std::endl;
        std::cout << "==========================================\n" << std::endl;

        return true;
    }

    // 检测函数实现
    bool BPU_Detect::Detect(const cv::Mat& input_img, cv::Mat& output_img) {
        if(!is_initialized_) {
            std::cout << "Please initialize first!" << std::endl;
            return false;
        }
        
        // 定义所有时间变量
        float preprocess_time = 0.0f;
        float infer_time = 0.0f;
        float postprocess_time = 0.0f;
        float draw_time = 0.0f;
        float total_time = 0.0f;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
    #if ENABLE_DRAW
        input_img.copyTo(output_img);
    #endif

        bool success = true;
        
        // 预处理
        {
            auto preprocess_start = std::chrono::high_resolution_clock::now();
            success = PreProcess(input_img);
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(
                preprocess_end - preprocess_start).count() / 1000.0f;
            
            if (!success) {
                std::cout << "Preprocess failed" << std::endl;
                goto cleanup;  
            }
        }
        
        // 推理
        {
            auto infer_start = std::chrono::high_resolution_clock::now();
            success = Inference();
            auto infer_end = std::chrono::high_resolution_clock::now();
            infer_time = std::chrono::duration_cast<std::chrono::microseconds>(
                infer_end - infer_start).count() / 1000.0f;
            
            if (!success) {
                std::cout << "Inference failed" << std::endl;
                goto cleanup;
            }
        }
        
        // 后处理
        {
            auto postprocess_start = std::chrono::high_resolution_clock::now();
            success = PostProcess();
            auto postprocess_end = std::chrono::high_resolution_clock::now();
            postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(
                postprocess_end - postprocess_start).count() / 1000.0f;
            
            if (!success) {
                std::cout << "Postprocess failed" << std::endl;
                goto cleanup;
            }
        }
        
        // 绘制结果
        {
            auto draw_start = std::chrono::high_resolution_clock::now();
            DrawResults(output_img);
            auto draw_end = std::chrono::high_resolution_clock::now();
            draw_time = std::chrono::duration_cast<std::chrono::microseconds>(
                draw_end - draw_start).count() / 1000.0f;
        }
        
        // 计算总时间
        {
            auto total_end = std::chrono::high_resolution_clock::now();
            total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                total_end - total_start).count() / 1000.0f;
        }
        
        // 打印时间统计
        std::cout << "\n============ Time Statistics ============" << std::endl;
        std::cout << "Preprocess time: " << std::fixed << std::setprecision(2) << preprocess_time << " ms" << std::endl;
        std::cout << "Inference time: " << std::fixed << std::setprecision(2) << infer_time << " ms" << std::endl;
        std::cout << "Postprocess time: " << std::fixed << std::setprecision(2) << postprocess_time << " ms" << std::endl;
        std::cout << "Draw time: " << std::fixed << std::setprecision(2) << draw_time << " ms" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " ms" << std::endl;
        std::cout << "FPS: " << std::fixed << std::setprecision(2) << 1000.0f / total_time << std::endl;
        std::cout << "======================================\n" << std::endl;

    cleanup:
        // 清理资源
        if (task_handle_) {
            hbDNNReleaseTask(task_handle_);
            task_handle_ = nullptr;
        }
        
        // 释放输入内存
        if(input_tensor_.sysMem[0].virAddr) {
            hbSysFreeMem(&(input_tensor_.sysMem[0]));
            input_tensor_.sysMem[0].virAddr = nullptr;
        }
        
        return success;
    }

    // 预处理实现
    bool BPU_Detect::PreProcess(const cv::Mat& input_img) {
        // 使用letterbox方式进行预处理
        x_scale_ = std::min(1.0f * input_h_ / input_img.rows, 1.0f * input_w_ / input_img.cols);
        y_scale_ = x_scale_;
        
        int new_w = input_img.cols * x_scale_;
        x_shift_ = (input_w_ - new_w) / 2;
        int x_other = input_w_ - new_w - x_shift_;
        
        int new_h = input_img.rows * y_scale_;
        y_shift_ = (input_h_ - new_h) / 2;
        int y_other = input_h_ - new_h - y_shift_;
        
        cv::resize(input_img, resized_img_, cv::Size(new_w, new_h));
        cv::copyMakeBorder(resized_img_, resized_img_, y_shift_, y_other, 
                        x_shift_, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
        
        // 转换为NV12格式
        cv::Mat yuv_mat;
        cv::cvtColor(resized_img_, yuv_mat, cv::COLOR_BGR2YUV_I420);
        
        // 准备输入tensor
        hbSysAllocCachedMem(&input_tensor_.sysMem[0], int(3 * input_h_ * input_w_ / 2));
        uint8_t* yuv = yuv_mat.ptr<uint8_t>();
        uint8_t* ynv12 = (uint8_t*)input_tensor_.sysMem[0].virAddr;
        // 计算UV部分的高度和宽度，以及Y部分的大小
        int uv_height = input_h_ / 2;
        int uv_width = input_w_ / 2;
        int y_size = input_h_ * input_w_;
        // 将Y分量数据复制到输入张量
        memcpy(ynv12, yuv, y_size);
        // 获取NV12格式的UV分量位置
        uint8_t* nv12 = ynv12 + y_size;
        uint8_t* u_data = yuv + y_size;
        uint8_t* v_data = u_data + uv_height * uv_width;
        // 将U和V分量交替写入NV12格式
        for(int i = 0; i < uv_width * uv_height; i++) {
            *nv12++ = *u_data++;
            *nv12++ = *v_data++;
        }
        // 将内存缓存清理，确保数据准备好可以供模型使用
        hbSysFlushMem(&input_tensor_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);// 清除缓存，确保数据同步
        return true;
    }

    // 推理实现
    bool BPU_Detect::Inference() {
        // 确保先释放之前的任务
        if (task_handle_) {
            hbDNNReleaseTask(task_handle_);
            task_handle_ = nullptr;
        }
        
        // 释放之前的输出内存
        for(int i = 0; i < 3; i++) {
            if(output_tensors_ && output_tensors_[i].sysMem[0].virAddr) {
                hbSysFreeMem(&(output_tensors_[i].sysMem[0]));
                output_tensors_[i].sysMem[0].virAddr = nullptr;
            }
        }
        
        // 初始化输入tensor属性
        input_tensor_.properties = input_properties_;
        
        // 获取输出tensor属性并分配内存
        for(int i = 0; i < 3; i++) {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, i),
                "Get output tensor properties failed");
            output_tensors_[i].properties = output_properties;
            
            // 为输出分配内存
            int out_aligned_size = output_properties.alignedByteSize;
            RDK_CHECK_SUCCESS(
                hbSysAllocCachedMem(&output_tensors_[i].sysMem[0], out_aligned_size),
                "Allocate output memory failed");
        }
        
        // 设置推理控制参数
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
        
        // 执行推理
        RDK_CHECK_SUCCESS(
            hbDNNInfer(&task_handle_, &output_tensors_, &input_tensor_, dnn_handle_, &infer_ctrl_param),
            "Model inference failed");
        
        // 等待任务完成
        RDK_CHECK_SUCCESS(
            hbDNNWaitTaskDone(task_handle_, 0),
            "Wait task done failed");
        
        return true;
    }

    // 后处理实现
    bool BPU_Detect::PostProcess() {
        detections_.clear();
        nms_indices_.clear();

        // 对齐 infer_onnx.py：输出顺序与 anchor 顺序分别为 20x20, 40x40, 80x80
        ProcessFeatureMap(output_tensors_[output_order_[0]], H_32, W_32, l_anchors_);
        ProcessFeatureMap(output_tensors_[output_order_[1]], H_16, W_16, m_anchors_);
        ProcessFeatureMap(output_tensors_[output_order_[2]], H_8, W_8, s_anchors_);

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        boxes.reserve(detections_.size());
        confidences.reserve(detections_.size());
        for (const auto& det : detections_) {
            boxes.push_back(det.box);
            confidences.push_back(det.confidence);
        }

        cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, nms_indices_, 1.f, nms_top_k_);
        
        return true;
    }

    // 打印检测结果实现
    void BPU_Detect::PrintResults() const {
        std::cout << "\n============ Detection Results ============" << std::endl;
        std::cout << "Total detections: " << nms_indices_.size() << std::endl;

        for (size_t i = 0; i < nms_indices_.size(); ++i) {
            int idx = nms_indices_[i];
            if (idx < 0 || idx >= static_cast<int>(detections_.size())) {
                continue;
            }
            const auto& det = detections_[idx];
            const std::string& color_name = (det.color_id >= 0 && det.color_id < static_cast<int>(color_names_.size()))
                                                ? color_names_[det.color_id]
                                                : std::string("unknown_color");
            const std::string& class_name = (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size()))
                                                ? class_names_[det.class_id]
                                                : std::string("unknown_class");

            std::cout << "  " << color_name << " " << class_name
                << ", confidence=" << std::fixed << std::setprecision(2) << det.confidence
                << ", bbox4pts=[(" << det.polygon[0].x << ", " << det.polygon[0].y << "), "
                << "(" << det.polygon[1].x << ", " << det.polygon[1].y << "), "
                << "(" << det.polygon[2].x << ", " << det.polygon[2].y << "), "
                << "(" << det.polygon[3].x << ", " << det.polygon[3].y << ")]"
                << std::endl;
        }
        std::cout << "========================================\n" << std::endl;
    }

    // 绘制结果实现
    void BPU_Detect::DrawResults(cv::Mat& img) {
    #if ENABLE_DRAW
        int overlay_y = 20;
        for (size_t i = 0; i < nms_indices_.size(); ++i) {
            int idx = nms_indices_[i];
            if (idx < 0 || idx >= static_cast<int>(detections_.size())) {
                continue;
            }
            const auto& det = detections_[idx];

            std::vector<cv::Point> polygon(4);
            for (int p = 0; p < 4; ++p) {
                polygon[p] = cv::Point(static_cast<int>(det.polygon[p].x), static_cast<int>(det.polygon[p].y));
            }

            cv::polylines(img, polygon, true, cv::Scalar(0, 255, 0), static_cast<int>(line_size_));
            for (const auto& pt : polygon) {
                cv::circle(img, pt, 1, cv::Scalar(0, 255, 0), 1);
            }

            cv::Point2f centroid(0.0f, 0.0f);
            for (const auto& pt : det.polygon) {
                centroid.x += pt.x;
                centroid.y += pt.y;
            }
            centroid.x /= 4.0f;
            centroid.y /= 4.0f;

            const std::string& color_name = (det.color_id >= 0 && det.color_id < static_cast<int>(color_names_.size()))
                                                ? color_names_[det.color_id]
                                                : std::string("unknown_color");
            const std::string& class_name = (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size()))
                                                ? class_names_[det.class_id]
                                                : std::string("unknown_class");

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << det.confidence;
            std::string text = color_name + " " + class_name + ": " + oss.str();

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
            int text_x = std::max(0, static_cast<int>(centroid.x) - text_size.width / 2);
            int text_y = std::max(15, static_cast<int>(centroid.y) - 10);

            cv::putText(img, text, cv::Point(text_x, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

            std::ostringstream bbox_ss;
            bbox_ss << "#" << (i + 1) << " ["
                    << static_cast<int>(det.polygon[0].x) << "," << static_cast<int>(det.polygon[0].y) << "] ["
                    << static_cast<int>(det.polygon[1].x) << "," << static_cast<int>(det.polygon[1].y) << "] ["
                    << static_cast<int>(det.polygon[2].x) << "," << static_cast<int>(det.polygon[2].y) << "] ["
                    << static_cast<int>(det.polygon[3].x) << "," << static_cast<int>(det.polygon[3].y) << "]";
            if (overlay_y < img.rows - 10) {
                cv::putText(
                    img, bbox_ss.str(), cv::Point(10, overlay_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                overlay_y += 16;
            }
        }
    #endif
        // 打印检测结果
        PrintResults();
    }

    // 特征图处理辅助函数
    void BPU_Detect::ProcessFeatureMap(hbDNNTensor& output_tensor, 
                                    int height, int width,
                                    const std::vector<std::pair<double, double>>& anchors) {
        // 检查量化类型
        if (output_tensor.properties.quantiType != NONE) {
            std::cout << "Output tensor quantization type should be NONE!" << std::endl;
            return;
        }
        
        // 刷新内存
        hbSysFlushMem(&output_tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        
        // 获取输出数据指针
        auto* raw_data = reinterpret_cast<float*>(output_tensor.sysMem[0].virAddr);

        const auto& shape = output_tensor.properties.validShape;
        if (shape.numDimensions != 4) {
            std::cout << "Unexpected output dims: " << shape.numDimensions << std::endl;
            return;
        }

        const int d1 = shape.dimensionSize[1];
        const int d2 = shape.dimensionSize[2];
        const int d3 = shape.dimensionSize[3];

        const bool is_nhwc = (d1 == height && d2 == width && d3 == static_cast<int>(anchors.size()) * OUTPUT_FEATURE_DIM);
        const bool is_nchw = (d1 == static_cast<int>(anchors.size()) * OUTPUT_FEATURE_DIM && d2 == height && d3 == width);
        if (!is_nhwc && !is_nchw) {
            std::cout << "Output shape mismatch, got ("
                    << shape.dimensionSize[0] << ", " << d1 << ", " << d2 << ", " << d3
                    << "), expected NHWC (1," << height << "," << width << "," << static_cast<int>(anchors.size()) * OUTPUT_FEATURE_DIM
                    << ") or NCHW (1," << static_cast<int>(anchors.size()) * OUTPUT_FEATURE_DIM << "," << height << "," << width << ")"
                    << std::endl;
            return;
        }

        auto get_value = [&](int anchor_idx, int feat_idx, int h_idx, int w_idx) -> float {
            if (is_nhwc) {
                const int c_idx = anchor_idx * OUTPUT_FEATURE_DIM + feat_idx;
                const int index = ((h_idx * width + w_idx) * (static_cast<int>(anchors.size()) * OUTPUT_FEATURE_DIM)) + c_idx;
                return raw_data[index];
            }
            const int c_idx = anchor_idx * OUTPUT_FEATURE_DIM + feat_idx;
            const int index = ((c_idx * height + h_idx) * width) + w_idx;
            return raw_data[index];
        };
        
        const float stride = static_cast<float>(input_h_) / static_cast<float>(height);

        // 遍历特征图的每个位置
        for(int h = 0; h < height; h++) {
            for(int w = 0; w < width; w++) {
                for(size_t anchor_idx = 0; anchor_idx < anchors.size(); ++anchor_idx) {
                    const auto& anchor = anchors[anchor_idx];

                    float confidence = Sigmoid(get_value(static_cast<int>(anchor_idx), 8, h, w));
                    if (confidence < score_threshold_) {
                        continue;
                    }

                    int color_id = 0;
                    for (int i = 1; i < COLOR_CLASS_COUNT; ++i) {
                        if (get_value(static_cast<int>(anchor_idx), 9 + i, h, w) >
                            get_value(static_cast<int>(anchor_idx), 9 + color_id, h, w)) {
                            color_id = i;
                        }
                    }

                    int class_id = 0;
                    for (int i = 1; i < CLASS_CLASS_COUNT; ++i) {
                        if (get_value(static_cast<int>(anchor_idx), 13 + i, h, w) >
                            get_value(static_cast<int>(anchor_idx), 13 + class_id, h, w)) {
                            class_id = i;
                        }
                    }

                    const float cls_score = Sigmoid(get_value(static_cast<int>(anchor_idx), 13 + class_id, h, w));
                    confidence *= cls_score;
                    if (confidence < score_threshold_) {
                        continue;
                    }

                    std::array<cv::Point2f, 4> points;
                    for (int pt = 0; pt < 4; ++pt) {
                        float decoded_x = get_value(static_cast<int>(anchor_idx), pt * 2, h, w) * static_cast<float>(anchor.first)
                                        + static_cast<float>(w) * stride;
                        float decoded_y = get_value(static_cast<int>(anchor_idx), pt * 2 + 1, h, w) * static_cast<float>(anchor.second)
                                        + static_cast<float>(h) * stride;
                        points[pt] = cv::Point2f(
                            (decoded_x - static_cast<float>(x_shift_)) / x_scale_,
                            (decoded_y - static_cast<float>(y_shift_)) / y_scale_);
                    }

                    std::array<cv::Point2f, 4> polygon = {points[0], points[3], points[2], points[1]};
                    float min_x = polygon[0].x;
                    float max_x = polygon[0].x;
                    float min_y = polygon[0].y;
                    float max_y = polygon[0].y;
                    for (int p = 1; p < 4; ++p) {
                        min_x = std::min(min_x, polygon[p].x);
                        max_x = std::max(max_x, polygon[p].x);
                        min_y = std::min(min_y, polygon[p].y);
                        max_y = std::max(max_y, polygon[p].y);
                    }

                    int box_w = static_cast<int>(max_x - min_x);
                    int box_h = static_cast<int>(max_y - min_y);
                    if (box_w <= 0 || box_h <= 0) {
                        continue;
                    }

                    DetectionResult result;
                    result.box = cv::Rect(static_cast<int>(min_x), static_cast<int>(min_y), box_w, box_h);
                    result.confidence = confidence;
                    result.class_id = class_id;
                    result.color_id = color_id;
                    result.polygon = polygon;
                    detections_.push_back(result);
                }
            }
        }
    }

    // 释放资源实现
    bool BPU_Detect::Release() {
        if(!is_initialized_) {
            return true;
        }
        
        // 释放任务
        if(task_handle_) {
            hbDNNReleaseTask(task_handle_);
            task_handle_ = nullptr;
        }
        
        try {
            // 释放输入内存
            if(input_tensor_.sysMem[0].virAddr) {
                hbSysFreeMem(&(input_tensor_.sysMem[0]));
            }
            
            // 释放输出内存
            for(int i = 0; i < 3; i++) {
                if(output_tensors_ && output_tensors_[i].sysMem[0].virAddr) {
                    hbSysFreeMem(&(output_tensors_[i].sysMem[0]));
                }
            }
            
            if(output_tensors_) {
                delete[] output_tensors_;
                output_tensors_ = nullptr;
            }
            
            // 释放模型
            if(packed_dnn_handle_) {
                hbDNNRelease(packed_dnn_handle_);
                packed_dnn_handle_ = nullptr;
            }
        } catch(const std::exception& e) {
            std::cout << "Exception during release: " << e.what() << std::endl;
        }
        
        is_initialized_ = false;
        return true;
    }

    // 修改main函数
    int main() {
        // 创建检测器实例
        BPU_Detect detector;
        
        // 初始化
        if (!detector.Init()) {
            std::cout << "Failed to initialize detector" << std::endl;
            return -1;
        }

    #if DETECT_MODE == 0
        // 单张图片检测模式
        std::cout << "Single image detection mode" << std::endl;
        
        // 读取测试图片
        cv::Mat input_img = cv::imread(DEFAULT_SINGLE_IMAGE_PATH);
        if (input_img.empty()) {
            std::cout << "Failed to load image" << std::endl;
            return -1;
        }
        
        // 执行检测
        cv::Mat output_img;
    #if ENABLE_DRAW
        if (!detector.Detect(input_img, output_img)) {
            std::cout << "Detection failed" << std::endl;
            return -1;
        }
        // 保存结果
        cv::imwrite(DEFAULT_OUTPUT_IMAGE_PATH, output_img);
    #else
        if (!detector.Detect(input_img, output_img)) {
            std::cout << "Detection failed" << std::endl;
            return -1;
        }
    #endif

    #else
        // 实时检测模式
        std::cout << "Real-time detection mode" << std::endl;
        
        cv::VideoCapture cap(DEFAULT_VIDEO_PATH);
        if (!cap.isOpened()) {
            std::cout << "Failed to open video: " << DEFAULT_VIDEO_PATH << std::endl;
            return -1;
        }

        const int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double video_fps = cap.get(cv::CAP_PROP_FPS);
        if (video_fps <= 1e-3) {
            video_fps = 30.0;
        }

        cv::VideoWriter writer;
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(DEFAULT_OUTPUT_VIDEO_PATH, fourcc, video_fps, cv::Size(frame_w, frame_h));
        if (!writer.isOpened()) {
            std::cout << "Failed to open video writer: " << DEFAULT_OUTPUT_VIDEO_PATH << std::endl;
            return -1;
        }
        std::cout << "Saving result video to: " << DEFAULT_OUTPUT_VIDEO_PATH << std::endl;
        
        cv::Mat frame, output_frame;
        while (true) {
            // 读取一帧
            cap >> frame;
            if (frame.empty()) {
                std::cout << "Video finished or failed to read frame" << std::endl;
                break;
            }
            
            // 执行检测
            if (!detector.Detect(frame, output_frame)) {
                std::cout << "Detection failed" << std::endl;
                break;
            }

                writer.write(output_frame);
        }
        
    #if ENABLE_DRAW
        // 释放视频
        cap.release();
        cv::destroyAllWindows();
    #endif
        writer.release();
    #endif
        
        // 释放资源
        detector.Release();
        
        return 0;
    }

