#pragma once
#include "timeDay.h"
#include "timeHandler.h"

bool operator==(const Timehandler&, const TimeDay&);
bool operator==(const TimeDay&, const Timehandler&);
bool operator!=(const Timehandler&, const TimeDay&);
bool operator!=(const TimeDay&, const Timehandler&);
bool operator<(const Timehandler&, const TimeDay&);
bool operator<(const TimeDay&, const Timehandler&);
bool operator>(const Timehandler&, const TimeDay&);
bool operator>(const TimeDay&, const Timehandler&);