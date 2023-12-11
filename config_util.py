import configparser


Config = configparser.ConfigParser()
Config.read("CONFIG.ini")

def GetModelConfig(configName):
    return Config.get("SectionModelRuntime", configName)
