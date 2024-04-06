import 'package:auth_login/provider/lowkey_providers.dart';
import 'package:auth_login/screens/welcome_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

final theme = ThemeData(
  scaffoldBackgroundColor: const Color.fromARGB(255, 1, 0, 40),
  colorScheme: ColorScheme.fromSeed(
    brightness: Brightness.dark,
    seedColor: const Color.fromARGB(255, 16, 12, 136),
  ),
  textTheme: GoogleFonts.latoTextTheme(),
);

final lightTheme = ThemeData(
  scaffoldBackgroundColor: const Color.fromARGB(255, 116, 124, 156),
  colorScheme: ColorScheme.fromSeed(
    brightness: Brightness.dark,
    seedColor: const Color.fromARGB(255, 16, 12, 136),
  ),
  textTheme: GoogleFonts.latoTextTheme(),
);

void main() {
  runApp(
    const ProviderScope(
      child: MyApp(),
    ),
  );
}

class MyApp extends ConsumerWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp(
      themeAnimationCurve: Curves.bounceOut,
      theme: ref.watch(themeProvider),
      debugShowCheckedModeBanner: false,
      home: const WelcomeScreen(),
      //home: const WelcomeScreen(),
    );
  }
}
