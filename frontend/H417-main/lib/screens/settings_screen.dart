// ignore_for_file: use_build_context_synchronously

import 'dart:convert';

import 'package:auth_login/models/enums.dart';
import 'package:auth_login/provider/lowkey_providers.dart';
import 'package:auth_login/widgets/bright_button.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:http/http.dart' as http;

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key, required this.token});

  final String token;

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  String backgroundColorValue = "";
  String languageValue = Language.English.toString().split(".")[1];
  String truckColorValue = TruckColor.Yellow.toString().split(".")[1];

  bool notificationEnabled = false;
  String email = "";

  void changeLanguage(String value) {
    languageValue = value;
    setState(() {});
  }

  @override
  void initState() {
    super.initState();
    _getEmailFromToken();
    backgroundColorValue =
        ref.read(themeProvider).brightness.toString().split(".")[1];
  }

  void _getEmailFromToken() async {
    final response = await http.get(
      Uri.parse('https://auth.sunjet-project.de/auth/email'),
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer ${widget.token}",
      },
    );
    final jsonResp = jsonDecode(response.body);

    setState(() {
      email = jsonResp['email'] as String;
    });
  }

  void _launch() async {
    const url =
        "https://docs.google.com/forms/d/e/1FAIpQLSc1rQZwlGd5ZZgSL_Y_wFA8-5_NsPlFBqDoMwbyM92tx8NRvw/viewform";
    if (!await launchUrl(Uri.parse(url))) {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text("Error")));
    }
  }

  void changeTruckColor(String value) {
    truckColorValue = value;
    setState(() {});
  }

  void changeBackgroundColor(String value) {
    backgroundColorValue = value;
    if (backgroundColorValue ==
        BackgroundColors.Light.toString().split(".")[1]) {
      ref.read(themeProvider.notifier).update(
        (state) {
          return ThemeData(
            scaffoldBackgroundColor: const Color.fromARGB(255, 116, 124, 156),
            colorScheme: ColorScheme.fromSeed(
              brightness: Brightness.light,
              seedColor: const Color.fromARGB(255, 16, 12, 136),
            ),
            textTheme: GoogleFonts.latoTextTheme(),
          );
        },
      );
    } else if (backgroundColorValue ==
        BackgroundColors.Dark.toString().split(".")[1]) {
      ref.read(themeProvider.notifier).update(
        (state) {
          return ThemeData(
            scaffoldBackgroundColor: const Color.fromARGB(255, 1, 0, 40),
            colorScheme: ColorScheme.fromSeed(
              brightness: Brightness.dark,
              seedColor: const Color.fromARGB(255, 16, 12, 136),
            ),
            textTheme: GoogleFonts.latoTextTheme(),
          );
        },
      );
    }
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Center(
              child: Icon(
                Icons.person_outline,
                size: 70,
                color: Color.fromARGB(255, 167, 191, 221),
              ),
            ),
            const SizedBox(
              height: 10,
            ),
            Center(
              child: Text(
                email,
                style: GoogleFonts.kadwa(
                  color: const Color.fromARGB(255, 167, 191, 221),
                  fontSize: 24,
                ),
              ),
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Text(
                  'Language',
                  style: GoogleFonts.kadwa(
                    color: const Color(0xFFE2F1F3),
                    fontSize: 24,
                  ),
                  textAlign: TextAlign.center,
                ),
                const Spacer(),
                DropdownButton<String>(
                  hint: Container(
                    alignment: Alignment.center,
                    width: 130,
                    height: 50,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(10),
                      color: const Color(0xFFE2F1F3),
                    ),
                    child: Row(
                      children: [
                        const SizedBox(width: 10),
                        Text(
                          languageValue,
                          style: GoogleFonts.kadwa(
                            color: const Color(0xFF010028),
                          ),
                        ),
                        const Spacer(),
                        const Icon(
                          Icons.arrow_drop_down,
                          color: Color(0xFF010028),
                        ),
                      ],
                    ),
                  ),
                  underline: const SizedBox(
                    width: 20,
                    height: 10,
                  ),
                  dropdownColor: const Color(0xFFE2F1F3),
                  onChanged: (value) {
                    changeLanguage(value.toString());
                  },
                  icon: const SizedBox(width: 0.1),
                  items: [
                    DropdownMenuItem(
                      value: Language.English.toString().split(".")[1],
                      child: Text(
                        Language.English.toString().split(".")[1],
                        style: GoogleFonts.kadwa(
                          color: const Color(0xFF010028),
                          fontSize: 24,
                        ),
                      ),
                    ),
                    DropdownMenuItem(
                      value: Language.German.toString().split(".")[1],
                      child: Text(
                        Language.German.toString().split(".")[1],
                        style: GoogleFonts.kadwa(
                          color: const Color(0xFF010028),
                          fontSize: 24,
                        ),
                      ),
                    ),
                    DropdownMenuItem(
                      value: Language.French.toString().split(".")[1],
                      child: Text(
                        Language.French.toString().split(".")[1],
                        style: GoogleFonts.kadwa(
                          color: const Color(0xFF010028),
                          fontSize: 24,
                        ),
                      ),
                    ),
                  ],
                )
              ],
            ),
            const SizedBox(
              height: 20,
            ),
            Row(
              children: [
                Text(
                  'Truck color',
                  style: GoogleFonts.kadwa(
                    color: const Color(0xFFE2F1F3),
                    fontSize: 24,
                  ),
                  textAlign: TextAlign.center,
                ),
                const Spacer(),
                DropdownButton<String>(
                  hint: Container(
                    alignment: Alignment.center,
                    width: 130,
                    height: 50,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(10),
                      color: const Color(0xFFE2F1F3),
                    ),
                    child: Row(
                      children: [
                        const SizedBox(width: 10),
                        Text(
                          truckColorValue,
                          style: GoogleFonts.kadwa(
                            color: const Color(0xFF010028),
                          ),
                        ),
                        const Spacer(),
                        const Icon(
                          Icons.arrow_drop_down,
                          color: Color(0xFF010028),
                        ),
                      ],
                    ),
                  ),
                  underline: const SizedBox(
                    width: 20,
                    height: 10,
                  ),
                  dropdownColor: const Color(0xFFE2F1F3),
                  onChanged: (value) {
                    changeTruckColor(value.toString());
                  },
                  icon: const SizedBox(width: 0.1),
                  items: [
                    DropdownMenuItem(
                      value: TruckColor.Yellow.toString().split(".")[1],
                      child: Text(
                        TruckColor.Yellow.toString().split(".")[1],
                        style: GoogleFonts.kadwa(
                          color: const Color(0xFF010028),
                          fontSize: 24,
                        ),
                      ),
                    ),
                    DropdownMenuItem(
                      value: TruckColor.Red.toString().split(".")[1],
                      child: Text(
                        TruckColor.Red.toString().split(".")[1],
                        style: GoogleFonts.kadwa(
                          color: const Color(0xFF010028),
                          fontSize: 24,
                        ),
                      ),
                    ),
                    DropdownMenuItem(
                      value: TruckColor.Green.toString().split(".")[1],
                      child: Text(
                        TruckColor.Green.toString().split(".")[1],
                        style: GoogleFonts.kadwa(
                          color: const Color(0xFF010028),
                          fontSize: 24,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(
              height: 20,
            ),
            Row(
              children: [
                Text(
                  'Background color',
                  style: GoogleFonts.kadwa(
                    color: const Color(0xFFE2F1F3),
                    fontSize: 20,
                  ),
                  textAlign: TextAlign.center,
                ),
                const Spacer(),
                DropdownButton<String>(
                  hint: Container(
                    alignment: Alignment.center,
                    width: 100,
                    height: 50,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(10),
                      color: const Color(0xFFE2F1F3),
                    ),
                    child: Row(
                      children: [
                        const SizedBox(width: 10),
                        Text(
                          backgroundColorValue,
                          style: GoogleFonts.kadwa(
                            color: const Color(0xFF010028),
                          ),
                        ),
                        const Spacer(),
                        const Icon(
                          Icons.arrow_drop_down,
                          color: Color(0xFF010028),
                        ),
                      ],
                    ),
                  ),
                  underline: const SizedBox(
                    width: 20,
                    height: 10,
                  ),
                  dropdownColor: const Color(0xFFE2F1F3),
                  onChanged: (value) {
                    changeBackgroundColor(value.toString());
                  },
                  icon: const SizedBox(width: 0.1),
                  items: [
                    DropdownMenuItem(
                      value: BackgroundColors.Dark.toString().split(".")[1],
                      child: Text(
                        BackgroundColors.Dark.toString().split(".")[1],
                        style: GoogleFonts.kadwa(
                          color: const Color(0xFF010028),
                          fontSize: 24,
                        ),
                      ),
                    ),
                    DropdownMenuItem(
                      value: BackgroundColors.Light.toString().split(".")[1],
                      child: Text(
                        BackgroundColors.Light.toString().split(".")[1],
                        style: GoogleFonts.kadwa(
                          color: const Color(0xFF010028),
                          fontSize: 24,
                        ),
                      ),
                    ),
                    DropdownMenuItem(
                      value: BackgroundColors.System.toString().split(".")[1],
                      child: Text(
                        BackgroundColors.System.toString().split(".")[1],
                        style: GoogleFonts.kadwa(
                          color: const Color(0xFF010028),
                          fontSize: 24,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(
              height: 20,
            ),
            Row(
              children: [
                Text(
                  'Notifications',
                  style: GoogleFonts.kadwa(
                    color: const Color(0xFFE2F1F3),
                    fontSize: 24,
                  ),
                  textAlign: TextAlign.center,
                ),
                const Spacer(),
                Switch(
                  value: notificationEnabled,
                  onChanged: (value) {
                    notificationEnabled = value;
                    setState(() {});
                  },
                  activeColor: const Color(0xFFA7BFDD),
                ),
              ],
            ),
            const SizedBox(height: 30),
            const Divider(),
            const SizedBox(height: 30),
            Center(
              child: BrightButton(
                onTap: _launch,
                title: "Support",
              ),
            ),
          ],
        ),
      ),
    );
  }
}
