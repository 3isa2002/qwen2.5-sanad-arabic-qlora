"""Disjoint validation and held-out test scenarios for Sanad QLoRA evaluation."""

from __future__ import annotations

from src.data.schema import ScenarioSpec, TurnVariant

# Split-exclusive distractor pool for `val` (zero overlap with train or test).
VAL_DISTRACTORS: tuple[tuple[str, str], ...] = (
    (
        "policies",
        "إشعارات التخفيضات داخل تطبيق الهاتف يمكن تخصيصها حسب الفئة المفضلة من قائمة الإعدادات دون التأثير على تنبيهات الشحن.",
    ),
    (
        "orders",
        "يمكن تحميل ملخص الطلبات الشهري بصيغة CSV من لوحة تحكم المشتريات لحسابات الأفراد في أي وقت.",
    ),
    (
        "payments",
        "التحقق الثنائي عند الدفع بالبطاقات يتطلب إدخال رمز 3D Secure المرسل من البنك المُصدر خلال دقيقتين.",
    ),
    (
        "listings",
        "يسمح متجر Sanad بإضافة حتى 20 منتجاً في قائمة المقارنة السريعة لمراجعة المواصفات جنباً إلى جنب.",
    ),
    (
        "returns",
        "تغليف الكرتون المعاد تدويره يحمل ملصقاً بيئياً ولا يؤثر على إمكانية إرجاع المنتج داخل المهلة المقررة.",
    ),
    (
        "shipping",
        "يمكن للعميل إضافة عنوان عمل وعنوان منزل منفصلين في دفتر العناوين والاختيار بينهما قبل تأكيد الطلب.",
    ),
)

# Split-exclusive distractor pool for `test` (zero overlap with train or val).
TEST_DISTRACTORS: tuple[tuple[str, str], ...] = (
    (
        "policies",
        "سجل التصفح المحلي يعرض آخر 15 منتجاً تمت مشاهدتها ويمكن مسحه بضغطة واحدة من إعدادات الخصوصية.",
    ),
    (
        "shipping",
        "تتيح صفحة التتبع مشاركة رابط قراءة فقط مع أحد أفراد العائلة لمتابعة وقت وصول الشحنة دون تسجيل دخول.",
    ),
    (
        "payments",
        "أسعار المنتجات المعروضة في المتجر تُحدّث لحظياً بالجنيه المصري دون الحاجة لتحويل عملات يدوياً.",
    ),
    (
        "listings",
        "مقاسات الملابس المعروضة في جداول القياس تتبع القياس الأوروبي (EU) مع إرفاق ما يقابلها بالسنتيمتر.",
    ),
    (
        "account",
        "تغيير كلمة المرور يتطلب أن تتكون من 10 أحرف على الأقل وتتضمن رقماً ورمزاً خاصاً لتعزيز الأمان.",
    ),
    (
        "returns",
        "صندوق الشحن الخارجي الواقي يمكن فتحه لإخراج فاتورة التسليم دون الإضرار بالتغليف الداخلي للمنتج.",
    ),
)


VAL_SCENARIOS: tuple[ScenarioSpec, ...] = (
    # -------------------- VAL GROUNDED SINGLE --------------------
    ScenarioSpec(
        scenario_group="val_furniture_assembly",
        split="val",
        domain="shipping",
        behavior="grounded_single",
        supporting_chunks=(
            "توصيل الأثاث الكبير يشمل إدخال القطع للغرفة المطلوبة مجاناً، بينما يتكلف تركيب النجار أو الفني الاختياري 250 جنيهاً ويُجدول خلال 48 ساعة من الاستلام.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "لو اشتريت دولاب كبير، هل التوصيل بيكون لباب العمارة بس ولا بيطلعوه الشقة والتركيب بكام؟",
                "التوصيل بيشمل إدخال الأثاث للغرفة اللي تحددها مجاناً، ولو عاوز خدمة التركيب الاختيارية بتتكلف 250 جنيه وبتتجدول خلال 48 ساعة من الاستلام {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "خدمة تركيب الأثاث بتاخد وقت قد إيه بعد ما القطع توصل البيت؟",
                "تركيب النجار أو الفني الاختياري بيتكلف 250 جنيه وبيتجدول خلال 48 ساعة من استلام الأثاث {c0}.",
            ),
            TurnVariant(
                "en",
                "Does large furniture delivery bring items inside my apartment room, and how much is optional assembly?",
                "Yes, large furniture delivery includes free room-of-choice placement, while optional technician assembly costs 250 EGP and is scheduled within 48 hours of delivery {c0}.",
            ),
            TurnVariant(
                "en",
                "How soon after furniture delivery is optional assembly scheduled?",
                "Optional technician assembly is scheduled within 48 hours of delivery for a 250 EGP fee {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="val_bnpl_early_settlement",
        split="val",
        domain="payments",
        behavior="grounded_single",
        supporting_chunks=(
            "Early settlement on 12-month store financing waives all remaining monthly interest after month 3, subject to a flat 100 EGP administrative fee.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "لو بقسّط مع المتجر على 12 شهر وعاوز أسدّد بدري بعد الشهر الرابع، هدفع باقي الفوايد؟",
                "السداد المبكر على تمويل المتجر الـ 12 شهر بيعفيك من كل الفوايد الشهرية المتبقية بعد الشهر التالت، مع دفع رسم إداري ثابت 100 جنيه بس {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "كام المصاريف الإدارية للسداد المبكر لتمويل 12 شهر؟",
                "الرسوم الإدارية للسداد المبكر هي 100 جنيه ثابتة، وبيلغى باقي الفوايد الشهرية بعد الشهر التالت {c0}.",
            ),
            TurnVariant(
                "en",
                "If I settle a 12-month store financing plan early in month 5, are remaining interest charges waived?",
                "Yes, early settlement after month 3 waives all remaining monthly interest and carries only a flat 100 EGP administrative fee {c0}.",
            ),
            TurnVariant(
                "en",
                "What fee applies to early settlement on 12-month financing?",
                "A flat 100 EGP administrative fee applies, while remaining monthly interest after month 3 is waived {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="val_prescription_eyewear_return",
        split="val",
        domain="returns",
        behavior="grounded_single",
        supporting_chunks=(
            "العدسات الطبية المفصلة حسب مقاس النظر لا تقبل الإرجاع لتغيير الرأي، بينما تقبل إطارات النظارات غير المستخدمة بدون عدسات مفصلة الاستبدال خلال 7 أيام.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "فصلت نضارة طبية بعدسات نظري بس شكلها ما عجبنيش، ينفع أرجعها؟",
                "العدسات الطبية المفصلة حسب مقاس النظر ما بتتقبلش في الاسترجاع لتغيير الرأي؛ الاستبدال متاح بس للإطارات غير المستخدمة من غير عدسات مفصلة خلال 7 أيام {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو طلبت شنبر نضارة فاضي من غير عدسات، أقدر أبدله خلال كام يوم؟",
                "تقدر تبدّل إطارات النظارات غير المستخدمة اللي بدون عدسات مفصلة خلال 7 أيام {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I return custom prescription eyeglasses if I simply don't like how they look?",
                "No, custom-ground prescription lenses cannot be returned for buyer's remorse; only unused frames without custom lenses qualify for a 7-day exchange {c0}.",
            ),
            TurnVariant(
                "en",
                "How many days do I have to exchange an unused eyeglass frame without custom lenses?",
                "Unused frames without custom lenses can be exchanged within 7 days {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="val_two_factor_recovery",
        split="val",
        domain="account",
        behavior="grounded_single",
        supporting_chunks=(
            "استعادة الحساب عند ضياع شريحة الهاتف تتطلب رفع صورة وجه البطاقة الشخصية مع حجب الرقم القومي ما عدا آخر 4 أرقام وصورة سيلفي وتُراجع خلال 24 إلى 48 ساعة عمل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "الخط بتاعي ضاع ومش عارف أستلم كود التحقق، أسترجع الحساب إزاي وهياخد وقت قد إيه؟",
                "تقدر تسترجع حسابك برفع صورة وش البطاقة الشخصية مع إخفاء الرقم القومي ما عدا آخر 4 أرقام وصورة سيلفي، والطلب بيتراجع خلال 24 لـ 48 ساعة عمل {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل لازم أسيب الرقم القومي كله ظاهر في صورة البطاقة لاستعادة الحساب؟",
                "لأ، بالعكس مطلوب حجب الرقم القومي ما عدا آخر 4 أرقام مع رفع صورة سيلفي، والمراجعة بتاخد 24 إلى 48 ساعة عمل {c0}.",
            ),
            TurnVariant(
                "en",
                "I lost my SIM card and cannot receive 2FA codes. How do I recover my account?",
                "Upload the front of your national ID (masking all digits except the last 4) together with a selfie; verification takes 24 to 48 business hours {c0}.",
            ),
            TurnVariant(
                "en",
                "How long does lost-SIM account recovery review take?",
                "Lost-SIM recovery reviews take 24 to 48 business hours after submitting your masked ID front and selfie {c0}.",
            ),
        ),
    ),

    # -------------------- VAL GROUNDED MULTI --------------------
    ScenarioSpec(
        scenario_group="val_cold_chain_groceries",
        split="val",
        domain="shipping",
        behavior="grounded_multi",
        supporting_chunks=(
            "المنتجات المبردة والطازجة تُشحن حصراً داخل القاهرة والإسكندرية عبر سيارات مبردة مخصصة.",
            "يجب الإبلاغ عن أي منتج مبرد تالف أو فاسد بالصور خلال 4 ساعات فقط من الاستلام للحصول على رصيد محفظة فوري.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "ممكن أطلب جبن ومنتجات مبردة تتبعت طنطا؟ ولو وصلت سايحة أو بايظة أبلغ في خلال قد إيه؟",
                "للأسف المنتجات المبردة والطازجة بتتشحن جوه القاهرة وإسكندرية بس بسيارات مبردة ومش بتوصل طنطا {c0}. وبصفة عامة لو وصل منتج مبرد فاسد أو تالف لازم الإبلاغ بالصور خلال 4 ساعات من الاستلام عشان ينزلك رصيد محفظة فوري {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "أنا في القاهرة واستلمت أوردر مبرد لقيت فيه حاجة بايظة، معايا كام ساعة أبلغ؟",
                "لازم تبلغ بالصور خلال 4 ساعات فقط من الاستلام عشان تاخد رصيد محفظة فوري {c1}.",
            ),
            TurnVariant(
                "en",
                "Do chilled items ship outside Cairo and Alexandria, and what is the spoilage report window?",
                "Chilled and fresh items ship exclusively inside Cairo and Alexandria via refrigerated vans {c0}, and any spoiled or damaged items must be reported with photos within 4 hours of delivery for instant wallet credit {c1}.",
            ),
            TurnVariant(
                "en",
                "How quickly do I get compensated for spoiled cold-chain groceries if reported with photos within 4 hours?",
                "Reporting spoiled or damaged chilled items with photos within 4 hours of delivery earns instant store wallet credit {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="val_warranty_tv_onsite",
        split="val",
        domain="warranty",
        behavior="grounded_multi",
        supporting_chunks=(
            "التلفزيونات مقاس 43 بوصة فأكثر تحصل على زيارة فني ضمان مجانية بالمنزل بدلاً من نقل الشاشة للفرع.",
            "احتراق الشاشة الثابت (Screen Burn-in) الناتج عن تشغيل صورة ثابتة بسطوع أعلى من 90% مستثنى من تغطية الضمان.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "التليفزيون عندي 55 بوصة وفيه عيب ضمان، لازم أشيله وأوديه التوكيل؟ وهل الـ Burn-in متغطي؟",
                "التلفزيونات 43 بوصة فأكثر بيجيلها فني صيانة مجاني في البيت بدل ما تنقلها للفرع {c0}؛ لكن لو المشكلة Screen Burn-in بسبب صورة ثابتة على سطوع فوق 90%، فده مستثنى من الضمان {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "شاشة 32 بوصة ولا 50 بوصة اللي بيجيلها فني صيانة في البيت مجاناً؟",
                "الزيارة المنزلية المجانية لفني الضمان متاحة للتلفزيونات مقاس 43 بوصة فأكثر {c0}.",
            ),
            TurnVariant(
                "en",
                "Do I have to carry my 50-inch TV to a service center, and does warranty cover screen burn-in?",
                "TVs 43 inches and larger receive free on-site technician visits instead of drop-off {c0}; however, screen burn-in caused by static images above 90% brightness is excluded from warranty {c1}.",
            ),
            TurnVariant(
                "en",
                "Which TV screen sizes qualify for free home warranty technician visits?",
                "TVs measuring 43 inches and larger qualify for free at-home warranty visits {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="val_seller_ads_credits",
        split="val",
        domain="listings",
        behavior="grounded_multi",
        supporting_chunks=(
            "تكلفة رصيد إعلانات ترويج القوائم للبائعين تبلغ 2 جنيه لكل نقرة موثقة وتنتهي صلاحية الرصيد بعد 90 يوماً من الشحن.",
            "إلغاء الحملة الإعلانية المبكر يرد الرصيد غير المستخدم إلى محفظة إعلانات البائع فقط ولا يُصرف نقداً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "النقرة في إعلانات البائعين بكام، ولو لغيت الحملة بدري الفلوس بترجعلي كاش في البنك؟",
                "تكلفة النقرة الموثقة 2 جنيه وصلاحية الرصيد 90 يوم من الشحن {c0}، ولو لغيت الحملة بدري الرصيد المتبقي بيرجع لمحفظة الإعلانات بس ومش بيتصرف كاش {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "رصيد إعلانات الترويج بيفضل شغال كام يوم قبل ما ينتهي؟",
                "صلاحية رصيد إعلانات ترويج القوائم هي 90 يوماً من تاريخ الشحن، وتكلفة النقرة الموثقة 2 جنيه {c0}.",
            ),
            TurnVariant(
                "en",
                "How much is each verified ad click for sellers, and can cancelled ad credits be cashed out?",
                "Sponsored listing ads cost 2 EGP per verified click with credits expiring after 90 days {c0}. Unused credits from cancelled campaigns return to the seller's ad wallet only and cannot be withdrawn as cash {c1}.",
            ),
            TurnVariant(
                "en",
                "Where does unused budget go if I cancel a seller ad campaign early?",
                "Cancelled campaigns refund unspent budget exclusively back to your seller ad credits balance rather than cash {c1}.",
            ),
        ),
    ),

    # -------------------- VAL HANDOFF --------------------
    ScenarioSpec(
        scenario_group="val_ho_trap_tv_bracket_install",
        split="val",
        domain="warranty",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "التلفزيونات مقاس 43 بوصة فأكثر تحصل على زيارة فني ضمان مجانية بالمنزل بدلاً من نقل الشاشة للفرع.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "هل الفني اللي بيجي مع التليفزيون الـ 55 بوصة بيخرّم الحيطة ويركّب الحامل الجداري وقت التسليم وبكام؟",
                "HANDOFF | السياسة المتاحة بتغطي زيارة فني الضمان للعيوب فقط ومفيهاش تفاصيل عن خدمة تركيب الحامل الجداري أو سعرها، وهحوّل حضرتك لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "بتوفروا تركيب وتوصيل وصلات الدش والريسيفر مع الشاشة؟",
                "HANDOFF | تفاصيل خدمات التركيب المنزلي الإضافية غير متوفرة في المستندات الحالية، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "en",
                "Does the home technician also drill and mount wall brackets for a newly delivered 55-inch TV, and what does it cost?",
                "HANDOFF | Wall-bracket installation services and pricing are not covered in the available warranty chunks. Let me transfer you to a support specialist.",
            ),
            TurnVariant(
                "en",
                "Is an extra VESA wall mount bracket included inside the 55-inch TV box?",
                "HANDOFF | Specific box contents for individual TV models are not included in the provided policies. Transferring you to an agent.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="val_ho_live_bnpl_rejection",
        split="val",
        domain="payments",
        behavior="handoff",
        handoff_subtype="account_state",
        supporting_chunks=(),
        anchor_chunks=(
            "Early settlement on 12-month store financing waives all remaining monthly interest after month 3, subject to a flat 100 EGP administrative fee.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "قدمت بالبطاقة على تمويل المتجر الـ 12 شهر والطلب اترفض النهارده، ممكن تعرفولي سبب الرفض الائتماني؟",
                "HANDOFF | مراجعة أسباب رفض التقييم الائتماني الشخصي مش متاحة عندي، وهحوّل حضرتك فوراً لموظف خدمة العملاء المختص يتابع معاك.",
            ),
            TurnVariant(
                "ar_eg",
                "عاوز أعرف إجمالي المبلغ المتبقي عليا بالظبط عشان أعمل سداد مبكر لخطة التمويل رقم #FN-310.",
                "HANDOFF | معرفة الرصيد الفعلي المتبقي على خطة #FN-310 يتطلب الدخول على حسابك التمويلي، وهحوّلك فوراً لموظف مختص.",
            ),
            TurnVariant(
                "en",
                "My application for 12-month store financing was declined today—can you tell me the exact credit reason?",
                "HANDOFF | I cannot view personal credit assessment results or rejection reasons. Let me transfer you to a human financing specialist.",
            ),
            TurnVariant(
                "en",
                "What is the exact payoff balance left on my financing plan #FN-310?",
                "HANDOFF | Checking the live payoff balance on plan #FN-310 requires access to your personal account. Transferring you now.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="val_ho_missing_corporate_credit_line",
        split="val",
        domain="listings",
        behavior="handoff",
        handoff_subtype="missing_fact",
        supporting_chunks=(),
        anchor_chunks=(
            "تكلفة رصيد إعلانات ترويج القوائم للبائعين تبلغ 2 جنيه لكل نقرة موثقة وتنتهي صلاحية الرصيد بعد 90 يوماً من الشحن.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عندكم باقة إعلانات للشركات الكبيرة بحد ائتماني آجل 500 ألف جنيه في الشهر؟",
                "HANDOFF | تفاصيل عقود الإعلانات المخصصة والحدود الائتمانية للشركات مش موجودة في المستندات المتاحة، وهحوّل حضرتك لفريق حسابات الشركات.",
            ),
            TurnVariant(
                "ar_eg",
                "إزاي أربط حساب الترويج بتاعي مع Meta Pixel أو Google Ads؟",
                "HANDOFF | أدلة التكامل التقني مع منصات الإعلانات الخارجية مش متوفرة عندي، وهحوّلك فوراً لموظف الدعم الفني.",
            ),
            TurnVariant(
                "en",
                "How do enterprise brands apply for a 500,000 EGP monthly postpaid ad credit line?",
                "HANDOFF | Enterprise postpaid ad credit lines are not documented in the available policies. Let me transfer you to a specialist.",
            ),
            TurnVariant(
                "en",
                "Do you offer conversion pixel tracking integrations for seller ad campaigns?",
                "HANDOFF | Technical ad tracking integrations are not covered in the provided chunks. Transferring you to support.",
            ),
        ),
    ),
)


TEST_SCENARIOS: tuple[ScenarioSpec, ...] = (
    # -------------------- TEST GROUNDED SINGLE --------------------
    ScenarioSpec(
        scenario_group="test_fragile_doorstep_inspection",
        split="test",
        domain="orders",
        behavior="grounded_single",
        supporting_chunks=(
            "طلبات الأدوات المنزلية الخزفية والزجاجية تتضمن مهلة انتظار إلزامية للمندوب مدتها 3 دقائق عند الباب لفحص أي كسر قبل التوقيع.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "طالب طقم صيني وخايف المندوب يمشي بسرعة، ليّا وقت قد إيه أفتحه وأبص على الكسر عند الباب؟",
                "طلبات الأدوات المنزلية الخزفية والزجاجية ليها مهلة انتظار إلزامية للمندوب 3 دقايق عند الباب عشان تفحص أي كسر قبل ما توقّع {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "المندوب مستعجل ومش عاوز يستنى أفتحه أتأكد من الطقم البورسلين، إيه السياسة؟",
                "من حقك المندوب يستنى مهلة إلزامية مدتها 3 دقائق عند الباب لفحص الأدوات الخزفية والزجاجية من أي كسر قبل التوقيع {c0}.",
            ),
            TurnVariant(
                "en",
                "How long must the courier wait at my door while I check ceramic dinnerware for breakage?",
                "Ceramic and glass homeware deliveries include a mandatory 3-minute doorstep courier wait window to inspect for breakage before signing {c0}.",
            ),
            TurnVariant(
                "en",
                "Am I allowed to check glass homeware at the door before signing the delivery slip?",
                "Yes, couriers are required to wait 3 minutes at the door so you can inspect glass or ceramic homeware for cracks prior to signing {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="test_gift_card_validity",
        split="test",
        domain="payments",
        behavior="grounded_single",
        supporting_chunks=(
            "بطاقات الهدايا الرقمية صالحة لمدة 12 شهراً من تاريخ التفعيل، ولا تُرد نقداً، ويمكن استهلاك رصيدها جزئياً على أكثر من طلب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "جالي كارت هدية بـ 1000 جنيه، مدته قد إيه وهل لازم أصرفه كله في أوردر واحد؟",
                "بطاقة الهدية الرقمية صالحة لمدة 12 شهر من تاريخ التفعيل، وتقدر تصرف رصيدها على أجزاء في أكتر من طلب، بس مش بتترد كاش {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "ينفع أستبدل رصيد كارت الهدية بفلوس كاش في إيدي أو على الكارت؟",
                "لأ يا فندم، بطاقات الهدايا الرقمية لا تُرد نقداً، لكن تقدر تستخدم رصيدها جزئياً على كذا طلب خلال 12 شهر من التفعيل {c0}.",
            ),
            TurnVariant(
                "en",
                "Do digital gift cards expire, and do I have to spend the entire balance in one checkout?",
                "Digital gift cards remain valid for 12 months from activation and support partial redemption across multiple orders, though they are non-refundable for cash {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I cash out an unused digital gift card balance?",
                "No, digital gift cards cannot be refunded for cash, but you can spend the balance across multiple orders within 12 months of activation {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="test_counterfeit_report_sla",
        split="test",
        domain="listings",
        behavior="grounded_single",
        supporting_chunks=(
            "البلاغات عن المنتجات المقلدة تجمد الإعلان خلال 6 ساعات لحين تقديم فواتير المورد الأصلي، وعند ثبوت التقليد يُعوض المشتري بضعف المبلغ المدفوع.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "بلغت عن إعلان بيبيع منتج مضروب ومقلد، بيتوقف إمتى ولو ثبت التقليد المشتري بياخد إيه؟",
                "بلاغات المنتجات المقلدة بتجمد الإعلان خلال 6 ساعات لحد ما البائع يقدم فواتير المورد الأصلي، ولو ثبت التقليد المشتري بيتعوّض بضعف المبلغ المدفوع {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "تعويض المشتري لو اكتشف إنه اشترى منتج مقلد وثبت ده كام؟",
                "عند ثبوت التقليد بعد فحص الفواتير، المشتري بيحصل على تعويض بضعف المبلغ المدفوع {c0}.",
            ),
            TurnVariant(
                "en",
                "How fast is a reported counterfeit listing frozen, and what compensation does a buyer receive if confirmed fake?",
                "Reported counterfeit listings are frozen within 6 hours pending original supplier invoices; if confirmed counterfeit, the buyer receives double (2x) the paid amount {c0}.",
            ),
            TurnVariant(
                "en",
                "What happens when someone reports a fake branded item listing?",
                "The listing is frozen within 6 hours for supplier invoice review, and confirmed counterfeit purchases earn buyers a 2x refund {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="test_account_gdpr_deletion",
        split="test",
        domain="account",
        behavior="grounded_single",
        supporting_chunks=(
            "يتم حذف الحساب نهائياً خلال 14 يوماً من تقديم الطلب إذا لم تكن هناك طلبات أو نزاعات مفتوحة، مع إخفاء البيانات الشخصية والاحتفاظ بالفواتير الضريبية طبقاً للقانون.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "لو قدمت طلب مسح حسابي نهائياً، بيتم إمتى وهل الفواتير القديمة بتتمسح هي كمان؟",
                "الحذف النهائي للحساب بيتم خلال 14 يوم من الطلب لو مفيش طلبات أو نزاعات مفتوحة، وبيتم إخفاء بياناتك الشخصية بس الفواتير الضريبية بتفضل محفوظة طبقاً للقانون {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "عندي طلب لسه ما وصلش وطلبت حذف الحساب، هيتحذف على طول؟",
                "لأ، الحذف النهائي بيتم خلال 14 يوم بشرط عدم وجود طلبات جارية أو نزاعات مفتوحة {c0}.",
            ),
            TurnVariant(
                "en",
                "How long does permanent account deletion take and are tax invoices also erased?",
                "Permanent deletion completes within 14 days provided there are no open orders or disputes; personal profile data is anonymized, while tax invoices are retained as required by law {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I permanently delete my account while a dispute is still open?",
                "No, permanent account deletion only proceeds after all open orders and disputes are closed, taking 14 days from the request {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="test_b2b_bulk_tier",
        split="test",
        domain="policies",
        behavior="grounded_single",
        supporting_chunks=(
            "Verified B2B accounts ordering 10 or more identical office units qualify for an automatic 7% wholesale tier discount and Net-15 invoice terms after credit approval.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "إحنا حساب B2B موثق وهنطلب 12 شاشة مكتب نفس الموديل، لينا خصم جملة وسداد آجل؟",
                "حسابات الـ B2B الموثقة لما تطلب 10 وحدات مكتبية متطابقة فأكتر بتاخد خصم جملة تلقائي 7% ونظام سداد آجل Net-15 بعد موافقة الائتمان {c0}.",
            ),
            TurnVariant(
                "en",
                "What quantity unlocks the automatic B2B wholesale discount and Net-15 terms?",
                "Verified B2B accounts ordering 10 or more identical office units receive a 7% wholesale discount automatically plus Net-15 terms upon credit approval {c0}.",
            ),
            TurnVariant(
                "en",
                "Is credit approval needed for Net-15 B2B invoicing?",
                "Yes, Net-15 invoice terms for 10+ identical office unit orders apply after credit approval for verified B2B accounts {c0}.",
            ),
        ),
    ),

    # -------------------- TEST GROUNDED MULTI --------------------
    ScenarioSpec(
        scenario_group="test_preorder_import_delay",
        split="test",
        domain="orders",
        behavior="grounded_multi",
        supporting_chunks=(
            "مواعيد وصول طلبات الاستيراد المسبق تقديرية، وإذا تأخر التخليص الجمركي لأكثر من 10 أيام عمل إضافية يحق للعميل الإلغاء الكامل دون خصم.",
            "إذا ضم الطلب منتجات متوفرة بالمخزن وأخرى حجزاً مسبقاً، تُشحن المنتجات المتوفرة أولاً ويتبعها الحجز المسبق لاحقاً بدون رسوم شحن إضافية.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "طلبت ماوس متوفر وكيبورد استيراد مسبق في أوردر واحد، هيوصلوا سوا؟ ولو الكيبورد اتأخر في الجمارك أسبوعين ألغيه؟",
                "المنتجات المتوفرة في المخزن بتتشحنلك الأول والكيبورد هيتبعها بعدين من غير رسوم شحن زيادة {c1}. ولو تأخير الجمارك عدى 10 أيام عمل إضافية عن الموعد التقديري، من حقك الإلغاء الكامل من غير أي خصم {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "حجزت شاشة مستوردة واتأخرت في الجمارك 13 يوم عمل زيادة عن المعاد، ينفع ألغي من غير غرامة؟",
                "أيوه، طالما التأخير في التخليص الجمركي زاد عن 10 أيام عمل إضافية يحقلك الإلغاء الكامل بدون أي خصم {c0}.",
            ),
            TurnVariant(
                "en",
                "If my cart mixes in-stock items and an imported pre-order, do they ship separately and when can I cancel customs delays?",
                "In-stock items dispatch first and the pre-order ships later at no extra shipping cost {c1}. If customs clearance delays the pre-order beyond 10 extra business days past the estimate, you may cancel in full without penalty {c0}.",
            ),
            TurnVariant(
                "en",
                "Will I pay two shipping fees when in-stock items ship ahead of a pre-order?",
                "No, the in-stock portion ships first and the pre-order follows separately with zero additional shipping fees {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="test_appliance_trade_in",
        split="test",
        domain="returns",
        behavior="grounded_multi",
        supporting_chunks=(
            "برنامج استبدال الأجهزة القديمة العاملة يخصم قيمة التقييم المبدئي كقسيمة فورية من سعر شراء الغسالة أو الثلاجة الجديدة.",
            "إذا كشف الفحص عند الباب أن موتور الجهاز القديم تالف أو مفقود الأجزاء، يُلغى التقييم ويُخيّر العميل بين سداد فرق القسيمة أو إلغاء الطلب الجديد.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "هبدّل غسالة قديمة بجديدة عندكم، لو المندوب شاف عند الباب إن الموتور بايظ بيحصل إيه في القسيمة؟",
                "البرنامج بيخصم قيمة التقييم كقسيمة من سعر الغسالة الجديدة لو الجهاز القديم شغال {c0}؛ لكن لو الفحص عند الباب لقَى الموتور تالف أو أجزاء ناقصة، التقييم بيتلغي وتختار تدفع فرق القسيمة أو تلغي الطلب الجديد {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "إزاي بيشتغل برنامج استبدال الثلاجات والغسالات القديمة؟",
                "قيمة تقييم جهازك القديم الشغال بتتخصم كقسيمة فورية من تمن الجهاز الجديد {c0}، وعند التسليم بيتفحص الجهاز عند الباب ولو طلع الموتور بايظ أو ناقص أجزاء تقدر تدفع الفرق أو تلغي الطلب {c1}.",
            ),
            TurnVariant(
                "en",
                "How does appliance trade-in work and what happens if the doorstep check finds a dead motor?",
                "Working old washers or fridges earn an instant voucher deducted from the new appliance checkout {c0}. If doorstep inspection discovers a dead motor or missing parts, the trade-in is cancelled and you can either pay the voucher difference or cancel the new order {c1}.",
            ),
            TurnVariant(
                "en",
                "Am I forced to keep the new washer if my old unit fails the doorstep trade-in check?",
                "No, if the trade-in is voided due to a dead motor or missing parts, you may choose either to pay the voucher difference or cancel the new order {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="test_warranty_laptop_spill",
        split="test",
        domain="warranty",
        behavior="grounded_multi",
        supporting_chunks=(
            "Laptops sold directly by Sanad carry a 24-month local warranty covering motherboard and display hardware faults.",
            "The optional Accidental Spill Protection add-on covers one liquid incident per year with a fixed 1,200 EGP customer deductible.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "ضمان اللابتوب اللي بيتباع مباشرة من Sanad كام شهر؟ ولو شربت قهوة عليه وأنا مشترك في حماية السوائل بدفع كام؟",
                "اللابتوبات المباعة مباشرة من Sanad عليها ضمان محلي 24 شهر للوحة الأم والشاشة ضد الأعطال {c0}، ولو مشترك في إضافة حماية السوائل فهي بتغطي حادث سوائل واحد في السنة مع تحمّل العميل مبلغ ثابت 1200 جنيه {c1}.",
            ),
            TurnVariant(
                "en_to_ar",
                "إضافة حماية السوائل بتغطي كام حادثة في السنة؟",
                "حماية السوائل الاختيارية بتغطي حادثة سوائل واحدة كل سنة بمبلغ تحمّل ثابت 1200 جنيه {c1}.",
            ),
            TurnVariant(
                "en",
                "How long is the motherboard warranty on laptops sold directly by Sanad, and what is the spill plan deductible?",
                "Direct Sanad laptops include a 24-month local motherboard and display warranty {c0}, while the optional Accidental Spill Protection plan covers one liquid incident per year with a 1,200 EGP deductible {c1}.",
            ),
            TurnVariant(
                "en",
                "Does the Accidental Spill add-on cover unlimited liquid incidents per year?",
                "No, it covers one liquid incident per year with a fixed 1,200 EGP deductible {c1}.",
            ),
        ),
    ),

    # -------------------- TEST HANDOFF --------------------
    ScenarioSpec(
        scenario_group="test_ho_trap_tradein_laptops",
        split="test",
        domain="returns",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "برنامج استبدال الأجهزة القديمة العاملة يخصم قيمة التقييم المبدئي كقسيمة فورية من سعر شراء الغسالة أو الثلاجة الجديدة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "ينفع أستبدل لابتوب قديم أو بطارية عربية في برنامج الاستبدال عشان أشتري موبايل جديد؟",
                "HANDOFF | برنامج الاستبدال المذكور في السياسات بيخص الغسالات والتلاجات بس ومفيش معلومات عن استبدال اللابتوبات أو بطاريات السيارات، وهحوّلك لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن تقيمولي غسالة توشيبا 7 كيلو موديل 2018 هتديني قسيمة كام جنيه بالظبط؟",
                "HANDOFF | تسعير تقييم موديل معين مش متاح عندي في السياسات العامة، وهحوّل حضرتك لموظف خدمة العملاء يوصلك بأداة أو فريق التقييم.",
            ),
            TurnVariant(
                "en",
                "Can I trade in an old gaming laptop or car battery under your trade-in program?",
                "HANDOFF | The available policy only covers washing machines and refrigerators and does not specify laptops or car batteries. Let me transfer you to a support specialist.",
            ),
            TurnVariant(
                "en",
                "How much exact trade-in voucher value do you pay for a 2019 12-foot fridge?",
                "HANDOFF | Exact model-by-model trade-in quotes are not available in the policy documentation. Transferring you to an agent.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="test_ho_trap_gift_card_cashout",
        split="test",
        domain="payments",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "بطاقات الهدايا الرقمية صالحة لمدة 12 شهراً من تاريخ التفعيل، ولا تُرد نقداً، ويمكن استهلاك رصيدها جزئياً على أكثر من طلب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "معايا كارت هدية من 2024 صلاحيته خلصت إمبارح وفيه 400 جنيه، ممكن تمدوهولي شهر كاستثناء؟",
                "HANDOFF | طلبات تمديد صلاحية بطاقات الهدايا المنتهية كاستثناء بتحتاج مراجعة مشرف خدمة العملاء، وهحوّل حضرتك فوراً.",
            ),
            TurnVariant(
                "ar_eg",
                "الكارت الرقمي بتاعي اتسرق كوده من الإيميل قبل ما أفعّله، أقدر أوقفه وأطلع بدل فاقد؟",
                "HANDOFF | حالات فقدان أو سرقة أكواد بطاقات الهدايا وإيقافها بتحتاج تدخل فريق الأمان وخدمة العملاء، وهحوّلك حالاً.",
            ),
            TurnVariant(
                "en",
                "My digital gift card expired yesterday with 400 EGP left—can you grant a courtesy extension?",
                "HANDOFF | Courtesy extensions on expired gift cards require supervisor approval. Let me transfer you to a support agent.",
            ),
            TurnVariant(
                "en",
                "Can I transfer a gift card balance from my account to my brother's account?",
                "HANDOFF | Account-to-account gift card balance transfers are not detailed in the provided policy chunks. Transferring you to a human specialist.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="test_ho_live_counterfeit_ticket",
        split="test",
        domain="listings",
        behavior="handoff",
        handoff_subtype="account_state",
        supporting_chunks=(),
        anchor_chunks=(
            "البلاغات عن المنتجات المقلدة تجمد الإعلان خلال 6 ساعات لحين تقديم فواتير المورد الأصلي، وعند ثبوت التقليد يُعوض المشتري بضعف المبلغ المدفوع.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "قدمت بلاغ تقليد برقم #CF-9031 يوم التلات، التحقيق وصل لإيه وهل البائع قدم فواتيره؟",
                "HANDOFF | متابعة نتائج التحقيق الخاصة بالبلاغ #CF-9031 بتحتاج مراجعة ملف القضية على السيستم، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "تعويض الضعف للبلاغ #CF-9031 ما نزلش في حسابي ليه لحد دلوقتي؟",
                "HANDOFF | التحقق من حالة صرف التعويض الخاص بالبلاغ #CF-9031 بيتم عبر الفريق المختص على السيستم، وهحوّلك حالاً.",
            ),
            TurnVariant(
                "en",
                "What is the investigation status of my counterfeit report ticket #CF-9031 from Tuesday?",
                "HANDOFF | I cannot look up live case notes for counterfeit report #CF-9031. Let me transfer you to a support specialist to check the ticket.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="test_ho_missing_b2b_api_docs",
        split="test",
        domain="policies",
        behavior="handoff",
        handoff_subtype="missing_fact",
        supporting_chunks=(),
        anchor_chunks=(
            "Verified B2B accounts ordering 10 or more identical office units qualify for an automatic 7% wholesale tier discount and Net-15 invoice terms after credit approval.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عندنا نظام ERP في الشركة، فين مفاتيح الـ REST API عشان نربط أوامر الشراء الـ B2B أوتوماتيك؟",
                "HANDOFF | وثائق الربط البرمجي (REST API) ومفاتيح التكامل لأنظمة الـ ERP مش متاحة في مستندات خدمة العملاء عندي، وهحوّلك لفريق الدعم المؤسسي.",
            ),
            TurnVariant(
                "en",
                "Where can our engineering team obtain B2B REST API documentation and sandbox credentials for ERP procurement?",
                "HANDOFF | B2B API integration documentation and sandbox credentials are not covered in the available support chunks. Transferring you to our B2B technical team.",
            ),
            TurnVariant(
                "en",
                "Do you support SAP Ariba cXML punchout catalogs for verified B2B accounts?",
                "HANDOFF | SAP Ariba and cXML integration support is not mentioned in the provided documentation. Let me connect you with an enterprise specialist.",
            ),
        ),
    ),
)
