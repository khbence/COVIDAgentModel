package main

import (
	"statistics_creator/creator"
	"statistics_creator/utils"
)

func main() {
	utils.Init()
	data, err := creator.CreateConfigRandomData("../../inputFiles/KamiData/agents.json", "../../inputFiles/KamiData/locations.json", []creator.AgeInterval{
		{
			Begin: 0,
			End:   50,
		},
		{
			Begin: 50,
			End:   75,
		},
		{
			Begin: 75,
			End:   500,
		},
	})
	if err != nil {
		utils.ErrorLogger.Println(err)
		return
	}
	err = data.WriteToFile("output.json")
	if err != nil {
		utils.ErrorLogger.Println(err)
		return
	}
}
