def call_openai(model_name, prompt):
    """
    Call OpenAI's API with the given prompt using v0.28.0
    
    Args:
        model_name: Name of the OpenAI model to use
        prompt: Prompt to send to the model
        
    Returns:
        tuple: (response_text, prompt_tokens, completion_tokens, latency)
    """
    print(f"🔄 OPENAI API CALL STARTED: {model_name}")
    start_time = time.time()
    
    try:
        # Load API keys
        api_keys = load_api_keys()
        openai_key = api_keys.get('openai')
        
        if not openai_key:
            print(f"❌ ERROR: No OpenAI API key found in configuration")
            return "Error: No valid OpenAI API key found", 0, 0, 0
        
        print(f"🔍 DEBUG: OpenAI API Key (first 5 chars): {openai_key[:5]}...")
        
        # Set API key for OpenAI v0.28.0
        import openai
        openai.api_key = openai_key
        
        # Clean model name if needed
        model_short_name = model_name.replace('openai:', '')
        print(f"📤 Sending request to OpenAI API for model: {model_short_name}")
        
        # Validate the model name
        is_valid, suggested_model = validate_openai_model(model_short_name)
        if not is_valid and suggested_model:
            print(f"⚠️ WARNING: {model_short_name} is not a standard OpenAI model name.")
            print(f"🔄 Using suggested model: {suggested_model}")
            model_short_name = suggested_model
        
        # Construct the messages
        print(f"📝 Creating chat message with prompt: {prompt[:50]}...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes customer reviews."},
            {"role": "user", "content": prompt}
        ]
        
        # Make the API call using OpenAI v0.28.0
        print(f"🚀 Sending request to OpenAI API...")
        try:
            response = openai.ChatCompletion.create(
                model=model_short_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract response data
            response_text = response['choices'][0]['message']['content']
            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            
            print(f"✅ OPENAI API CALL SUCCESS: {model_short_name}")
            print(f"⏱️ Latency: {latency:.2f}s")
            print(f"🔤 Tokens: {prompt_tokens} prompt, {completion_tokens} completion")
            print(f"📝 Response first 100 chars: {response_text[:100]}...")
            
            return response_text, prompt_tokens, completion_tokens, latency
            
        except Exception as api_error:
            error_msg = str(api_error)
            print(f"❌ Error during API call: {error_msg}")
            return f"Error: API call failed: {error_msg}", 0, 0, time.time() - start_time
    
    except Exception as e:
        print(f"❌ OPENAI API CALL FAILED: {model_name}")
        print(f"⏱️ Failed after: {(time.time() - start_time):.2f}s")
        print(f"🚨 Error: {str(e)}")
        return f"Error: {str(e)}", 0, 0, time.time() - start_time
