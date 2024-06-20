def chatbot_response(user_input):
   
    # Normalize the input by converting it to lowercase
    user_input = user_input.lower()

    # Check for greetings
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"
    
    # Check for inquiry about the chatbot's state
    elif "how are you" in user_input:
        return "I'm just a bot, but I'm here to help you! How can I assist you?"

    # Check for inquiry about the chatbot's name
    elif "your name" in user_input:
        return "I'm Chatbot, your friendly assistant!"

    # Check if the user is asking for the current time
    elif "time" in user_input:
        from datetime import datetime
        # Get the current time and format it
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    
    # Check if the user is asking for the current date
    elif "date" in user_input:
        from datetime import datetime
        # Get the current date and format it
        return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
    
    # Check if the user is asking about the weather
    elif "weather" in user_input:
        return "I'm not connected to the internet, but I suggest checking a weather app for the latest updates."

    # Check if the user is asking about the chatbot's favorite color
    elif "favorite color" in user_input:
        return "As a bot, I don't have preferences, but I think blue is quite nice!"
    elif "Thinks makes us happy" or "feeling bad" in user_input:
        return "Listen to your favourite music, it will make you feel happy"

    # Check if the user is asking for a simple arithmetic operation
    elif "add" in user_input or "plus" in user_input:
        # Attempt to extract numbers from the input and perform addition
        try:
            numbers = [int(s) for s in user_input.split() if s.isdigit()]
            if len(numbers) == 2:
                return f"The result of adding {numbers[0]} and {numbers[1]} is {numbers[0] + numbers[1]}."
            else:
                return "Please provide two numbers to add."
        except ValueError:
            return "I couldn't understand the numbers you provided."

    # Check for farewell messages
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day!"

    # Default response for unrecognized input
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Main loop to interact with the chatbot
if __name__ == "__main__":
    user_input = input("You: ")  # Prompt the user for input
    while user_input.lower() not in ["bye", "goodbye"]:  # Continue until the user says goodbye
        response = chatbot_response(user_input)  # Get the chatbot's response
        print(f"Chatbot: {response}")  # Display the response
        user_input = input("You: ")  # Prompt the user for more input
    print("Chatbot: Goodbye! Have a great day!")  # Final farewell message
