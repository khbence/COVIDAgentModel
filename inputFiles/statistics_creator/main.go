package main

import (
	"statistics_creator/creator"
	"statistics_creator/utils"
)

func main() {
	utils.Init()
	data, err := creator.CreateConfigRandomData("files/agents.json", "files/locations.json", []creator.AgeInterval{
		{
			Begin: 0,
			End:   200,
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
