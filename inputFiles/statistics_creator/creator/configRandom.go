package creator

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"statistics_creator/utils"
	"strconv"
)

type chanceDict struct {
	Value  string  `json:"value"`
	Chance float64 `json:"chance"`
}

type chanceDictDiag struct {
	Value      string  `json:"value"`
	Chance     float64 `json:"chance"`
	DiagChance float64 `json:"diagnosedChance"`
}

type chanceSlice []chanceDict
type stateSlice []stateData
type ageIntervalSlice []AgeInterval
type chanceDictDiagSlice []chanceDictDiag

type stateData struct {
	AgeStart     int                 `json:"ageStart"`
	AgeEnd       int                 `json:"ageEnd"`
	Distribution chanceDictDiagSlice `json:"diagnosedChance"`
	agentCounter int
}

// ConfigRandomFormat stores the entire file
type ConfigRandomFormat struct {
	IrregularLocChance      float64     `json:"irregulalLocationChance"`
	LocationTypeDistibution chanceSlice `json:"locationTypeDistibution"`
	PreCondDistibution      chanceSlice `json:"preCondDistibution"`
	StateDistribution       stateSlice  `json:"stateDistibution"`
	AgentTypeDistribution   chanceSlice `json:"agentTypeDistribution"`
	agentCounter            int
	locationCounter         int
}

// AgeInterval is used to define what are the intervals to get the state distributions
type AgeInterval struct {
	Begin int
	End   int
}

func makeConfigRandomformat(ages []AgeInterval) ConfigRandomFormat {
	newCFG := ConfigRandomFormat{
		StateDistribution: make(stateSlice, len(ages)),
	}
	for idx := range newCFG.StateDistribution {
		newCFG.StateDistribution[idx].AgeStart = ages[idx].Begin
		newCFG.StateDistribution[idx].AgeEnd = ages[idx].End
	}
	return newCFG
}

func readJSON(filePath string) (map[string]interface{}, error) {
	result := make(map[string]interface{})
	file, err := os.Open(filePath)
	if err != nil {
		return result, err
	}
	defer file.Close()
	data, err := ioutil.ReadAll(file)
	if err != nil {
		return result, err
	}
	err = json.Unmarshal(data, &result)
	return result, err
}

func mapGet(dict map[string]interface{}, key string) interface{} {
	ret, ok := dict[key]
	if !ok {
		panic(fmt.Errorf("Key (%s) did not found in the file", key))
	}
	return ret
}

func mapGetString(dict map[string]interface{}, key string) string {
	tmp, ok := dict[key]
	if !ok {
		panic(fmt.Errorf("Key (%s) did not found in the file", key))
	}
	switch v := tmp.(type) {
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', 0, 64)
	default:
		panic(fmt.Errorf("Key (%s) could not transform directly or indirectly to string", key))
	}
}

func mapGetInt(dict map[string]interface{}, key string) int {
	tmp, ok := dict[key]
	if !ok {
		panic(fmt.Errorf("Key (%s) did not found in the file", key))
	}
	switch v := tmp.(type) {
	case float64:
		return int(v)
	case string:
		vInteger, err := strconv.Atoi(v)
		if err != nil {
			panic(fmt.Errorf("String to int conversion error for key (%s), error is : %v", key, err))
		}
		return vInteger
	default:
		panic(fmt.Errorf("Key (%s) could not transform directly or indirectly to int", key))
	}
}

func (pcd *chanceSlice) increment(key string) {
	found := false
	for i, value := range *pcd {
		if value.Value == key {
			(*pcd)[i].Chance += 1.0
			found = true
			break
		}
	}
	if !found {
		*pcd = append(*pcd, chanceDict{Value: key, Chance: 1.0})
	}
}

func (d *chanceDictDiagSlice) increment(state string) {
	found := false
	for i, value := range *d {
		if value.Value == state {
			(*d)[i].Chance++
			found = true
			break
		}
	}
	if !found {
		*d = append(*d, chanceDictDiag{
			Value:      state,
			Chance:     1.0,
			DiagChance: 0.0,
		})
	}
}

func (sd *stateSlice) increment(age int, state string) {
	for i, value := range *sd {
		if value.AgeStart < age && age < value.AgeEnd {
			(*sd)[i].agentCounter++
			(*sd)[i].Distribution.increment(state)
		}
	}
}

func (crf *ConfigRandomFormat) addPerson(person map[string]interface{}) {
	crf.agentCounter++
	crf.PreCondDistibution.increment(mapGetString(person, "preCond"))
	crf.StateDistribution.increment(mapGetInt(person, "age"), mapGetString(person, "state"))
	crf.AgentTypeDistribution.increment(mapGetString(person, "typeID"))
}

func (crf *ConfigRandomFormat) addLocation(location map[string]interface{}) {
	crf.locationCounter++
	crf.LocationTypeDistibution.increment(mapGetString(location, "type"))
}

// CreateConfigRandomData create configRandom data from the agents and locations JSON files
func CreateConfigRandomData(agentsFile string, locationFile string, ages []AgeInterval) (ConfigRandomFormat, error) {
	result := makeConfigRandomformat(ages)
	utils.InfoLogger.Println("Parsing agents file")
	agentsData, err := readJSON(agentsFile)
	if err != nil {
		return result, err
	}
	defer func() {
		if err := recover(); err != nil {
			utils.ErrorLogger.Println(err)
		}
	}()
	people := mapGet(agentsData, "people").([]interface{})
	for _, person := range people {
		result.addPerson(person.(map[string]interface{}))
	}

	utils.InfoLogger.Println("Parsing locations file")
	locationsData, err := readJSON(locationFile)
	if err != nil {
		return result, err
	}
	places := mapGet(locationsData, "places").([]interface{})
	for _, location := range places {
		result.addLocation(location.(map[string]interface{}))
	}

	return result, nil
}

func (pcd *chanceSlice) divideChances(number int) {
	numberF := float64(number)
	for i := range *pcd {
		(*pcd)[i].Chance /= numberF
	}
}

func (d *chanceDictDiagSlice) divideChances(number int) {
	numberF := float64(number)
	for i := range *d {
		(*d)[i].Chance /= numberF
	}
}

func (sd *stateSlice) calculatePercentages() {
	for i := range *sd {
		(*sd)[i].Distribution.divideChances((*sd)[i].agentCounter)
	}
}

func (crf *ConfigRandomFormat) calculatePercentages() {
	crf.AgentTypeDistribution.divideChances(crf.agentCounter)
	crf.LocationTypeDistibution.divideChances(crf.locationCounter)
	crf.PreCondDistibution.divideChances(crf.agentCounter)
	crf.StateDistribution.calculatePercentages()
}

// WriteToFile writes out it's data in JSON format to the file path
func (crf *ConfigRandomFormat) WriteToFile(file string) error {
	utils.InfoLogger.Println("Writing out file")
	crf.calculatePercentages()
	data, err := json.MarshalIndent(crf, "", "    ")
	if err != nil {
		return err
	}
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	defer f.Close()
	n, err := f.Write(data)
	if n != len(data) {
		return fmt.Errorf("Could not write every byte of data, for some reason")
	}
	return nil
}
