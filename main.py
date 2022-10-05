from utils.fbm_rrs import FbMarketing
from utils.price_prediction_lr import FbMarketingPricePrediction

if __name__ == '__main__':
    lr_cls = FbMarketingPricePrediction() 
    #lr_cls.main()

    cls = FbMarketing() 
    cls.main()
    