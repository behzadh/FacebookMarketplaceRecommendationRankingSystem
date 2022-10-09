from utils.category_prediction_svc import FbMarketingCategoryPrediction
from utils.price_prediction_lr import FbMarketingPricePrediction

if __name__ == '__main__':
    lr_cls = FbMarketingPricePrediction() 
    #lr_cls.main()

    cls = FbMarketingCategoryPrediction() 
    cls.main()
    