import requests
import os
from tqdm import tqdm
            
def main(date,km):
    # BBOX=（Bottom Left）min Latitude,min Longtitude（Top Right）max Latitude,max Longtitude
    url_form="https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME=%sT00:00:00Z&BBOX=%.4f,%.4f,%.4f,%.4f&CRS=EPSG:4326&LAYERS=VIIRS_SNPP_DayNightBand_At_Sensor_Radiance&WRAP=none&FORMAT=image/jpeg&WIDTH=255&HEIGHT=255"#"&ts=1687191158906"

    # 经度5km≈0.054°，纬度5km≈0.045°
    bias_Long=0.054/5*km #经度
    bias_La=0.045/5*km #纬度
    # bias_Long=bias_La=0.05/5*km
    

    path="./%s-%dkm/"%(date.replace("-",""),km)
    if not os.path.exists(path):
        os.mkdir(path)
        
    with open("loc.txt","r",encoding="utf-8") as f:
        with open("url_log.txt","a+",encoding="utf-8") as f3:
            for line in tqdm(f.readlines()):
                a,b,c=line.strip().split(" ")
                name=a+"-"+b
                Long,La=map(float,c.split("，"))
                url=url_form%(date,La-bias_La,Long-bias_Long,La+bias_La,Long+bias_Long)
                with open(path+name+".jpg","wb") as f2:
                    f2.write(requests.get(url).content)
                f3.write(name+"\t"+url+"\n")

if __name__=="__main__":
    for date in ["2023-05-24","2023-04-15","2023-06-08"]:
        for km in [5,10]:
            print(date,km,"km:")
            main(date,km)
            
            
# cv2.imread("1.jpg",flags=0)读取灰度图