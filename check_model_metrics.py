import joblib
import sys

def main():
    try:
        # 加载模型指标
        metrics = joblib.load('task3/models/model_metrics.joblib')
        
        print("评委得分模型:")
        print(f"  MSE: {metrics['judge_model']['mse']:.3f}")
        print(f"  RMSE: {metrics['judge_model']['rmse']:.3f}")
        print(f"  R^2: {metrics['judge_model']['r2']:.3f}")
        print(f"  交叉验证平均R^2: {metrics['judge_model']['avg_r2']:.3f}")
        print()
        print("粉丝投票模型:")
        print(f"  MSE: {metrics['fan_model']['mse']:.3f}")
        print(f"  RMSE: {metrics['fan_model']['rmse']:.3f}")
        print(f"  R^2: {metrics['fan_model']['r2']:.3f}")
        print(f"  交叉验证平均R^2: {metrics['fan_model']['avg_r2']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    main()