import 'package:auth_login/screens/about_us_screen.dart';
import 'package:auth_login/screens/crash_screen.dart';
import 'package:auth_login/screens/settings_screen.dart';
import 'package:flutter/material.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({
    super.key,
    required this.token,
  });

  final String token;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  late List<Widget> screenList;
  @override
  void initState() {
    super.initState();
    screenList = [
      SettingsScreen(
        token: widget.token,
      ),
      const CrashScreen(),
      const AboutUsScreen(),
    ];
  }

  int screenIndex = 1;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: screenList[screenIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: screenIndex,
        onTap: (value) {
          setState(() {
            screenIndex = value;
          });
        },
        items: const [
          BottomNavigationBarItem(
            label: "Settings",
            icon: Icon(Icons.settings_outlined),
            activeIcon: Icon(Icons.settings),
            backgroundColor: Colors.amber,
          ),
          BottomNavigationBarItem(
            label: "",
            icon: Icon(
              Icons.car_crash_outlined,
              size: 46,
            ),
            activeIcon: Icon(
              Icons.car_crash,
              size: 46,
            ),
            backgroundColor: Colors.green,
          ),
          BottomNavigationBarItem(
            label: "About us",
            icon: Icon(Icons.search_outlined),
            activeIcon: Icon(Icons.search),
            backgroundColor: Colors.red,
          ),
        ],
      ),
      appBar: AppBar(
        actions: [
          IconButton(
            onPressed: () {},
            icon: const Icon(
              Icons.help,
            ),
          )
        ],
        title: const Text("Deridefull"),
        toolbarHeight: 70,
        backgroundColor: Colors.transparent,
        leading: GestureDetector(
          onTap: () {
            Navigator.of(context).pop();
            Navigator.of(context).pop();
          },
          child: Hero(
            tag: "Image",
            child: Image.asset(
              'assets/image/logo_transparent.png',
            ),
          ),
        ),
      ),
    );
  }
}
