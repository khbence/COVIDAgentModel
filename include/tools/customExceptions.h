#pragma once
#include <exception>
#include <string>

class CustomErrors : public std::exception {
    std::string error;

protected:
    explicit CustomErrors(std::string&& error_p) : error(std::move(error_p)) {}

public:
    [[nodiscard]] const char* what() const noexcept override { return error.c_str(); }
};

namespace IOProgression {
    class TransitionInputError : public CustomErrors {
    protected:
        explicit TransitionInputError(std::string&& error_p)
            : CustomErrors("Transition input file error: " + error_p) {}
    };

    class WrongNumberOfStates : public TransitionInputError {
    public:
        WrongNumberOfStates(unsigned expected, unsigned got)
            : TransitionInputError("Expected " + std::to_string(expected) + " states, got "
                                   + std::to_string(got) + ".\n") {}
    };

    class WrongStateName : public TransitionInputError {
    public:
        explicit WrongStateName(const std::string& stateName)
            : TransitionInputError(stateName + " doesn't exists.\n") {}
    };

    class TooMuchBad : public TransitionInputError {
    public:
        explicit TooMuchBad(unsigned state)
            : TransitionInputError(
                std::to_string(state)
                + ". state has multiple bad transition, which is not allowed in this setup.\n") {}
    };

    class BadChances : public TransitionInputError {
    public:
        explicit BadChances(const std::string& state)
            : TransitionInputError("Sum of transition chances of state " + state + " is not 1.\n") {
        }
    };
}// namespace IOProgression

namespace IOParameters {
    class ParametersInputError : public CustomErrors {
    protected:
        explicit ParametersInputError(std::string&& error_p)
            : CustomErrors("Parameters input file error: " + error_p) {}
    };

    class NotBinary : public ParametersInputError {
    public:
        NotBinary() : ParametersInputError("There are two genders!!!\n") {}
    };

    class WrongGenderName : public ParametersInputError {
    public:
        WrongGenderName() : ParametersInputError("Define an F and an M genders.\n") {}
    };

    class NegativeFrom : public ParametersInputError {
    public:
        NegativeFrom()
            : ParametersInputError("From value in the age scaling should be positive value!\n") {}
    };
}// namespace IOParameters

namespace IOAgents {
    class AgentsInputError : public CustomErrors {
    protected:
        explicit AgentsInputError(std::string&& error_p)
            : CustomErrors("Agents input file error: " + error_p) {}
    };

    class InvalidGender : public AgentsInputError {
    public:
        explicit InvalidGender(const std::string& genderName)
            : AgentsInputError("Wronf gender name (" + genderName + ").\n") {}
    };

    class NotDefinedAge : public AgentsInputError {
    public:
        explicit NotDefinedAge(unsigned age)
            : AgentsInputError(
                "Age " + std::to_string(age) + " was not defined in the parameter input file!\n") {}
    };

    class NotDefinedCondition : public AgentsInputError {
    public:
        explicit NotDefinedCondition(unsigned conditionID)
            : AgentsInputError("Condition with ID " + std::to_string(conditionID)
                               + " was not defined in the parameter input file!\n") {}
    };
}// namespace IOAgents