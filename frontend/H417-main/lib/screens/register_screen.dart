// ignore_for_file: use_build_context_synchronously

import 'dart:convert';

import 'package:auth_login/widgets/bright_button.dart';
import 'package:auth_login/widgets/textfiled.dart';
import 'package:auth_login/widgets/transparent_button.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _form = GlobalKey<FormState>();
  String _enteredEmail = '';
  String _enteredPassword = '';

  void _register() async {
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
      Uri.parse("https://auth.sunjet-project.de/auth/register"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode(regBody),
    );
    ScaffoldMessenger.of(context).clearSnackBars();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          json.decode(response.body)['status']
              ? "Successfull registration"
              : "Registration failed",
        ),
      ),
    );
    if (json.decode(response.body)['status']) {
      return;
    }
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Form(
            key: _form,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'Registration',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.kadwa(
                    color: const Color(0xFFE6F0F3),
                    fontSize: 57,
                  ),
                ),
                const SizedBox(height: 50),
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
                  onValidate: (value) {
                    if (value == null || value.trim().length < 6) {
                      return "Password should have at least 6 characters long.";
                    }
                    return null;
                  },
                  hintText: 'Password',
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
                    BrightButton(onTap: _register, title: "Register")
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
