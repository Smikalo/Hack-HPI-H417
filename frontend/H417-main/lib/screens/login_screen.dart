// ignore_for_file: use_build_context_synchronously

import 'dart:convert';

import 'package:auth_login/screens/main_screen.dart';
import 'package:auth_login/widgets/bright_button.dart';
import 'package:auth_login/widgets/textfiled.dart';
import 'package:auth_login/widgets/transparent_button.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _form = GlobalKey<FormState>();
  // late SharedPreferences prefs;
  String _enteredEmail = '';
  String _enteredPassword = '';

  @override
  void initState() {
    super.initState();
    // initPref();
  }

  // void initPref() async {
  //   prefs = await SharedPreferences.getInstance();
  // }

  void _login() async {
    final isValid = _form.currentState!.validate();
    if (!isValid) {
      return;
    }

    _form.currentState!.save();

    var regBody = {
      "email": _enteredEmail,
      "password": _enteredPassword,
    };

    var response = await http.post(
      Uri.parse("https://auth.sunjet-project.de/auth/login"),
      headers: {
        "Content-Type": "application/json",
      },
      body: jsonEncode(regBody),
    );
    final jsonResponce = json.decode(response.body);
    ScaffoldMessenger.of(context).clearSnackBars();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          jsonResponce['status'] ? "Successfull login" : "Login failed",
        ),
      ),
    );
    if (!jsonResponce['status']) {
      return;
    }
    var myToken = jsonResponce['token'];
    // prefs.setString('token', myToken);
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (ctx) => MainScreen(
          token: myToken,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Form(
            key: _form,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                Hero(
                  tag: "Image",
                  child: Image.asset('assets/image/logo_transparent.png'),
                ),
                Text(
                  'Login',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.kadwa(
                    color: const Color(0xFFE6F0F3),
                    fontSize: 57,
                  ),
                ),
                const SizedBox(height: 20),
                CustomTextField(
                  onSaved: (p0) {
                    _enteredEmail = p0!;
                  },
                  onValidate: (value) {
                    if (value == null ||
                        value.trim().isEmpty ||
                        !value.contains("@")) {
                      return "Enter a valid email adress";
                    }
                    return null;
                  },
                  hintText: 'Email: example@tum.de',
                ),
                const SizedBox(height: 20),
                CustomTextField(
                  onSaved: (p0) {
                    _enteredPassword = p0!;
                  },
                  isPassword: true,
                  hintText: 'Password',
                  onValidate: (value) {
                    if (value == null || value.trim().length < 6) {
                      return "Password should have at least 6 characters long.";
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 50),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    TransparentButton(
                      onTap: () {
                        Navigator.of(context).pop();
                      },
                      title: "Cancel",
                    ),
                    BrightButton(onTap: _login, title: "Login")
                  ],
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}
