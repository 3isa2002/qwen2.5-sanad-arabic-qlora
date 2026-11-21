"""Training KB scenarios and split-exclusive distractor pool for Sanad QLoRA."""

from __future__ import annotations

from src.data.schema import ScenarioSpec, TurnVariant

# Disjoint distractor pool used only inside the `train` split.
# Tagged by domain so distractors never share the target scenario's domain.
TRAIN_DISTRACTORS: tuple[tuple[str, str], ...] = (
    (
        "policies",
        "يمكن للعملاء اختيار لغة الفاتورة وإشعارات البريد الإلكتروني (عربي أو إنجليزي) من تبويب التفضيلات في الملف الشخصي.",
    ),
    (
        "policies",
        "تحديث بيانات ملف المستخدم مثل الاسم أو البريد الإلكتروني يتم من صفحة إعدادات الحساب ويُحفظ فوراً بعد الضغط على تأكيد.",
    ),
    (
        "policies",
        "Wishlist collections can be shared via a read-only public link without exposing the owner's delivery address or phone number.",
    ),
    (
        "shipping",
        "تحديثات خط سير الشحنة تظهر كل ساعتين في صفحة التتبع بمجرد مسح الباركود في محطة الفرز الرئيسية.",
    ),
    (
        "shipping",
        "خزائن الاستلام الذاتية (Pickup Lockers) بتحتفظ بالطلبات المدفوعة مسبقاً لمدة 72 ساعة من وقت وصول رسالة الكود.",
    ),
    (
        "returns",
        "طباعة ملصق الإرجاع الإلكتروني ليست إلزامية للمندوب في القاهرة طالما تم تأكيد رقم طلب الاسترجاع عبر الرسالة النصية.",
    ),
    (
        "payments",
        "بطاقات ميزة والبطاقات البنكية المحلية المدعومة بخاصية 3D Secure مقبولة في الدفع الإلكتروني دون عمولات إضافية.",
    ),
    (
        "payments",
        "يمكن حفظ حتى 3 بطاقات دفع مشفرة داخل الحساب لتسريع إتمام الطلبات المستقبلية مع إمكانية حذفها في أي وقت.",
    ),
    (
        "warranty",
        "تسجيل الرقم التسلسلي للجهاز داخل صفحة ضماناتي يسهّل متابعة تاريخ انتهاء الضمان وتنزيل نسخة إلكترونية من الشهادة.",
    ),
    (
        "orders",
        "إضافة ملاحظة قصيرة لمندوب التوصيل مثل 'الاتصال عند البوابة' متاحة في خانة تعليمات العنوان قبل الدفع.",
    ),
    (
        "listings",
        "تقييمات المنتجات المكتوبة بعد التسليم الفعلي تحمل شارة 'شراء موثق' وتُراجع خلال 24 ساعة لمنع الألفاظ المسيئة.",
    ),
    (
        "account",
        "تفعيل التنبيهات عبر واتساب يتطلب الموافقة داخل إعدادات الإشعارات ويمكن إيقافه بضغطة واحدة في أي وقت.",
    ),
    (
        "promotions",
        "إلغاء الاشتراك في رسائل العروض الترويجية لا يؤثر على وصول رسائل تتبع الطلبات أو فواتير الشراء.",
    ),
    (
        "listings",
        "زر 'أبلغ عن مشكلة في السعر أو الوصف' متاح أسفل كل صفحة منتج لمراجعة دقة بيانات البائع من فريق الجودة.",
    ),
)


TRAIN_SCENARIOS: tuple[ScenarioSpec, ...] = (
    # -------------------- SHIPPING & DELIVERY --------------------
    ScenarioSpec(
        scenario_group="tr_ship_cairo_giza",
        split="train",
        domain="shipping",
        behavior="grounded_single",
        supporting_chunks=(
            "التوصيل داخل القاهرة والجيزة يتم خلال 24 إلى 48 ساعة عمل، وتبلغ رسوم الشحن 60 جنيهاً للطلبات أقل من 1000 جنيه.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "لو سمحت، الشحن جوه القاهرة بياخد كام يوم ومصاريفه كام؟",
                "التوصيل جوه القاهرة والجيزة بياخد من 24 لـ 48 ساعة عمل، ولو طلبك أقل من 1000 جنيه رسوم الشحن بتكون 60 جنيه {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "أنا ساكن في الجيزة وطلبت أوردر بـ 700 جنيه، هدفع شحن قد إيه وهيوصل إمتى؟",
                "بما إن الأوردر في الجيزة وأقل من 1000 جنيه، مصاريف الشحن هتكون 60 جنيه والتوصيل بيتم خلال 24 إلى 48 ساعة عمل {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هو التوصيل في المعادي بياخد نفس وقت الجيزة؟",
                "أيوه يا فندم، التوصيل داخل القاهرة والجيزة عموماً بيتم خلال 24 إلى 48 ساعة عمل، والرسوم 60 جنيه للطلبات اللي أقل من 1000 جنيه {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "كم تكلفة الشحن لأوردر قيمته 450 جنيه داخل القاهرة؟",
                "رسوم الشحن داخل القاهرة للطلبات اللي أقل من 1000 جنيه هي 60 جنيه، والتوصيل خلال 24 إلى 48 ساعة عمل {c0}.",
            ),
            TurnVariant(
                "en",
                "How long does standard delivery take inside Cairo and Giza, and what is the fee under 1000 EGP?",
                "Delivery within Cairo and Giza takes 24 to 48 business hours, with a 60 EGP shipping fee for orders under 1,000 EGP {c0}.",
            ),
            TurnVariant(
                "en",
                "I live in Giza and my cart is 800 EGP. How much is shipping?",
                "For orders under 1,000 EGP in Cairo and Giza, the shipping fee is 60 EGP and delivery takes 24 to 48 business hours {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ship_alex_delta",
        split="train",
        domain="shipping",
        behavior="grounded_single",
        supporting_chunks=(
            "Shipping to Alexandria and Delta governorates takes 2 to 3 business days at a flat rate of 75 EGP.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "الشحن لإسكندرية بياخد كام يوم وتكلفته قد إيه؟",
                "الشحن لإسكندرية ومحافظات الدلتا بياخد من 2 لـ 3 أيام عمل برسوم ثابتة 75 جنيه {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "لو أنا في المنصورة، المندوب هيوصلني في خلال قد إيه؟",
                "التوصيل لمحافظات الدلتا وإسكندرية بياخد من يومين لـ 3 أيام عمل، وسعر الشحن 75 جنيه {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "مصاريف شحن طنطا كام لو سمحت؟",
                "الشحن لمحافظات الدلتا وإسكندرية تكلفته 75 جنيه ثابتة، والتوصيل خلال 2 إلى 3 أيام عمل {c0}.",
            ),
            TurnVariant(
                "en",
                "What are the delivery lead time and shipping cost for Alexandria?",
                "Shipping to Alexandria and Delta governorates takes 2 to 3 business days for a flat 75 EGP fee {c0}.",
            ),
            TurnVariant(
                "en",
                "How fast can you deliver to Delta governorates?",
                "Deliveries to Alexandria and the Delta arrive within 2 to 3 business days at a flat fee of 75 EGP {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ship_upper_egypt",
        split="train",
        domain="shipping",
        behavior="grounded_single",
        supporting_chunks=(
            "التوصيل لمحافظات الصعيد من المنيا حتى أسوان يستغرق 3 إلى 5 أيام عمل مقابل رسوم شحن 95 جنيهاً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "بتوصلوا أسيوط وسوهاج؟ والشحن بياخد وقت قد إيه؟",
                "أيوه يا فندم، الشحن لمحافظات الصعيد من المنيا لحد أسوان بياخد من 3 لـ 5 أيام عمل ورسومه 95 جنيه {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "الشحن لأسوان بكام وبيوصل في كام يوم؟",
                "التوصيل لمحافظات الصعيد من المنيا إلى أسوان تكلفته 95 جنيه وبيستغرق من 3 إلى 5 أيام عمل {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "أنا من المنيا، الأوردر هيوصلني في خلال قد إيه؟",
                "الشحن لمحافظات الصعيد بداية من المنيا حتى أسوان بياخد من 3 لـ 5 أيام عمل بتكلفة 95 جنيه {c0}.",
            ),
            TurnVariant(
                "ar_to_en",
                "How much is shipping to Upper Egypt like Luxor or Aswan?",
                "Shipping to Upper Egypt governorates from Minya to Aswan costs 95 EGP and takes 3 to 5 business days {c0}.",
            ),
            TurnVariant(
                "ar_to_en",
                "What is the estimated delivery window for Minya?",
                "Deliveries to Upper Egypt (Minya through Aswan) take 3 to 5 business days with a 95 EGP shipping fee {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ship_free_threshold",
        split="train",
        domain="shipping",
        behavior="grounded_multi",
        supporting_chunks=(
            "يُمنح الشحن المجاني تلقائياً على الطلبات القياسية التي تتجاوز قيمتها 1000 جنيه مصري.",
            "الشحنات الثقيلة التي يتجاوز وزنها 15 كجم تخضع لرسوم شحن ثابتة قدرها 150 جنيهاً ولا يشملها عرض الشحن المجاني.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "طلبت ثلاجة ميني بار بـ 8000 جنيه ووزنها 22 كيلو، ليه اتحسب عليا 150 جنيه شحن ومفيش شحن مجاني؟",
                "الشحن المجاني بيتطبق على الطلبات القياسية اللي فوق 1000 جنيه بس {c0}، لكن الشحنات الثقيلة اللي وزنها أكتر من 15 كيلو بتتحاسب برسوم ثابتة 150 جنيه ومش بيشملها الشحن المجاني {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "إمتى باخد شحن مجاني، وهل العرض بيشمل الأجهزة الثقيلة؟",
                "الطلبات القياسية اللي قيمتها أكتر من 1000 جنيه بتاخد شحن مجاني تلقائي {c0}، لكن أي شحنة وزنها يتعدى 15 كجم لها رسوم شحن ثابتة 150 جنيه ومش داخلة في الشحن المجاني {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "الأوردر بتاعي 1500 جنيه ووزنه خفيف، هدفع شحن؟",
                "طالما طلبك قياسي ومش معدي 15 كيلو وقيمته فوق 1000 جنيه، هتاخد شحن مجاني تلقائياً {c0}؛ رسوم الـ 150 جنيه بتتفرض بس على الأوزان اللي فوق 15 كيلو {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "مكتوب شحن مجاني فوق 1000 جنيه بس اتضاف 150 جنيه في السلة على مايكروويف تقيل!",
                "عرض الشحن المجاني فوق 1000 جنيه مخصص للطلبات القياسية فقط {c0}، أما الشحنات اللي وزنها أكتر من 15 كجم فبيتحسب عليها رسوم ثابتة 150 جنيه وما بيشملهاش العرض {c1}.",
            ),
            TurnVariant(
                "en",
                "Why am I still charged 150 EGP shipping on an 18 kg heater that costs over 1,000 EGP?",
                "Free shipping applies automatically only to standard orders above 1,000 EGP {c0}, whereas heavy shipments exceeding 15 kg carry a flat 150 EGP freight fee and are excluded from free shipping {c1}.",
            ),
            TurnVariant(
                "en",
                "Does free shipping over 1000 EGP apply to items heavier than 15 kg?",
                "No. While standard orders over 1,000 EGP qualify for free shipping {c0}, items over 15 kg incur a mandatory 150 EGP fee and are excluded from free shipping {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ship_courier_reschedule",
        split="train",
        domain="shipping",
        behavior="grounded_single",
        supporting_chunks=(
            "يتواصل المندوب هاتفياً قبل الوصول بـ 30 دقيقة، وتُتاح محاولتا توصيل مجاناً قبل إرجاع الطلب تلقائياً إلى المستودع.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "لو المندوب جه وأنا مش في البيت، هيحصل إيه في الأوردر؟",
                "المندوب بيتصل بيك قبل ما يوصل بـ 30 دقيقة، ولو ما قدرتش تستلم أول مرة ليك محاولتين توصيل مجاناً قبل ما الطلب يرجع المستودع تلقائياً {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "المندوب بيرن قبلها بقد إيه؟ ولو فوتّ المكالمة الطلب بيتلغي على طول؟",
                "المندوب بيتواصل معاك قبل الوصول بـ 30 دقيقة، ومش بيتلغي فوراً لأن عندك محاولتي توصيل مجانيتين قبل ما يرجع المستودع {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "في كام محاولة تسليم متاحة للأوردر؟",
                "متاح لحضرتك محاولتا توصيل مجاناً، والمندوب بيكلمك قبل الوصول بـ 30 دقيقة {c0}.",
            ),
            TurnVariant(
                "en",
                "What happens if I miss the courier on the first delivery attempt?",
                "The courier calls 30 minutes before arrival, and you receive two free delivery attempts before the order is automatically returned to the warehouse {c0}.",
            ),
            TurnVariant(
                "en",
                "Does the driver call ahead before delivering?",
                "Yes, the courier calls 30 minutes prior to arrival, and two free delivery attempts are provided before any return to warehouse {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ship_pickup_lockers",
        split="train",
        domain="shipping",
        behavior="grounded_single",
        supporting_chunks=(
            "Prepaid orders routed to Sanad Pickup Lockers are held for 72 hours and require the SMS OTP code or original national ID for collection.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "طلبت استلام من الـ Pickup Locker، الأوردر بيفضل مستنيني قد إيه ومحتاج إيه وأنا بستلم؟",
                "الطلبات المدفوعة مسبقاً في خزائن الاستلام بتفضل محفوظة لمدة 72 ساعة، وبتحتاج كود الـ OTP اللي وصلك في رسالة أو البطاقة الشخصية الأصلية وقت الاستلام {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "ممكن أستلم من اللوكر بالبطاقة ولا لازم كود؟",
                "تقدر تستلم بكود الـ OTP المبعوت في رسالة SMS أو بالبطاقة الشخصية الأصلية، مع العلم إن الطلب بيفضل محجوز 72 ساعة بس {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "هل ينفع أختار استلام من اللوكر لو هدفع كاش؟",
                "خزائن الاستلام متاحة للطلبات المدفوعة مسبقاً فقط وبتتحجز 72 ساعة وتستلمها بكود الـ OTP أو البطاقة الشخصية {c0}.",
            ),
            TurnVariant(
                "en",
                "How long do pickup lockers hold an order and what do I need to collect it?",
                "Prepaid orders at pickup lockers are held for 72 hours and require either the SMS OTP code or your original national ID to collect {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I send someone with my OTP to pick up from a Sanad locker?",
                "Collection from a pickup locker requires the SMS OTP code or original national ID, and prepaid parcels are kept for up to 72 hours {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ship_weekend_cutoff",
        split="train",
        domain="shipping",
        behavior="grounded_multi",
        supporting_chunks=(
            "الطلبات المؤكدة قبل الساعة 2:00 ظهراً تُجهز وتُسلّم لشركة الشحن في نفس يوم العمل.",
            "أيام الجمعة والعطلات الرسمية لا تُحتسب ضمن أيام العمل ولا يتم فيها خروج شحنات جديدة من المستودع.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عملت أوردر يوم الخميس الساعة 4 العصر، هيخرج للشحن إمتى؟",
                "بما إن الطلب اتأكد بعد الساعة 2 ظهراً {c0}، ويوم الجمعة مش محسوب من أيام العمل ومبتخرجش فيه شحنات {c1}، فطلبك هيتجهز ويخرج في أول يوم عمل تالي.",
            ),
            TurnVariant(
                "ar_eg",
                "لو طلبت يوم الجمعة الصبح الشحن بيتحسب من نفس اليوم؟",
                "لأ يا فندم، أيام الجمعة والعطلات الرسمية مش بتتحسب ضمن أيام العمل ولا بتخرج فيها شحنات {c1}؛ التجهيز والتسليم لشركة الشحن في نفس اليوم بيكون للطلبات المؤكدة قبل 2 ظهراً في أيام العمل {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو أكدت الطلب الساعة 12 الظهر يوم الثلاثاء، هيتشحن النهارده؟",
                "أيوه، الطلبات اللي بتتأكد قبل الساعة 2 ظهراً بتتجهز وتتسلم لشركة الشحن في نفس يوم العمل {c0}، طالما اليوم ده مش جمعة أو إجازة رسمية {c1}.",
            ),
            TurnVariant(
                "en",
                "If I place an order at 3 PM on Thursday, why hasn't it shipped on Friday?",
                "Same-day dispatch only applies to orders confirmed before 2:00 PM on business days {c0}, and Fridays and public holidays are excluded from business days and warehouse dispatch {c1}.",
            ),
            TurnVariant(
                "en",
                "What is the daily dispatch cutoff and do you ship on Fridays?",
                "Orders confirmed before 2:00 PM dispatch the same business day {c0}, while Fridays and public holidays are not counted as business days and have no outgoing dispatches {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ship_same_day_cairo",
        split="train",
        domain="shipping",
        behavior="grounded_single",
        supporting_chunks=(
            "خدمة التوصيل السريع خلال 4 ساعات متاحة داخل القاهرة فقط للطلبات المؤكدة قبل 12:00 ظهراً برسوم 110 جنيهات.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عندكم توصيل سريع في نفس اليوم جوه القاهرة وبكام؟",
                "أيوه، متاح توصيل سريع خلال 4 ساعات جوه القاهرة بس لو أكدت الطلب قبل الساعة 12 ظهراً، وتكلفته 110 جنيه {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو طلبت توصيل 4 ساعات الساعة 2 الظهر في القاهرة هينفع؟",
                "لأ يا فندم، خدمة التوصيل خلال 4 ساعات داخل القاهرة بتحتاج تأكيد الطلب قبل 12:00 ظهراً، ورسومها 110 جنيه {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل خدمة الـ 4 ساعات السريعة متاحة لإسكندرية؟",
                "التوصيل السريع خلال 4 ساعات متاح داخل القاهرة فقط للطلبات المؤكدة قبل 12 ظهراً مقابل 110 جنيه {c0}.",
            ),
            TurnVariant(
                "en",
                "Do you offer express same-day delivery in Cairo?",
                "Yes, 4-hour express delivery is available inside Cairo only for orders confirmed before 12:00 PM at a fee of 110 EGP {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I get 4-hour express delivery if I order at 1 PM?",
                "No, the 4-hour Cairo-only express service requires order confirmation before 12:00 PM and costs 110 EGP {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ship_fragile_packaging",
        split="train",
        domain="shipping",
        behavior="grounded_multi",
        supporting_chunks=(
            "المنتجات الزجاجية والقابلة للكسر تُشحن بتغليف مزدوج مبطّن لحمايتها أثناء النقل.",
            "تصوير فيديو متصل لحظة فتح الكرتونة وإرسال البلاغ خلال 24 ساعة من الاستلام يسرّع الموافقة الفورية على تعويض الكسر.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "أنا طالب طقم كوبايات وخايف يتكسر في الشحن، بتأمنوه إزاي ولو وصل مكسور أعمل إيه؟",
                "المنتجات الزجاجية والقابلة للكسر بتتشحن بتغليف مزدوج مبطّن {c0}. ولو لا قدر الله وصل مكسور، صوّر فيديو متصل وقت فتح الكرتونة وابعت البلاغ خلال 24 ساعة عشان تاخد موافقة سريعة على التعويض {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "وصلني برطمان زجاج مكسور إمبارح، إيه الأوراق أو الإثبات المطلوب؟",
                "للموافقة السريعة على تعويض الكسر، بنحتاج فيديو متصل لحظة فتح الكرتونة وتقديم البلاغ خلال 24 ساعة من الاستلام {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل الزجاج بيتغلف بحماية إضافية؟",
                "أيوه، أي منتج زجاجي أو قابل للكسر بيتشحن بتغليف مزدوج مبطّن {c0}، وبرضه يُفضل تصوير فيديو متصل عند فتح الكرتونة تحسباً لأي كسر وتقديم البلاغ خلال 24 ساعة {c1}.",
            ),
            TurnVariant(
                "en",
                "How are fragile glassware items packed and what helps speed up breakage claims?",
                "Glass and fragile products are shipped in padded double-box packaging {c0}. Recording a continuous unboxing video and filing a report within 24 hours of delivery accelerates instant breakage approval {c1}.",
            ),
            TurnVariant(
                "en",
                "My glass item arrived cracked. What is the deadline to report with video?",
                "Please submit your breakage report along with a continuous unboxing video within 24 hours of receiving the parcel to speed up immediate approval {c1}.",
            ),
        ),
    ),

    # -------------------- RETURNS & REFUNDS --------------------
    ScenarioSpec(
        scenario_group="tr_ret_standard_14d",
        split="train",
        domain="returns",
        behavior="grounded_single",
        supporting_chunks=(
            "يمكن إرجاع المنتجات غير المستخدمة خلال 14 يوماً من تاريخ الاستلام بشرط بقاء التغليف الأصلي سليماً وإرفاق الفاتورة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "الأوردر وصلني إمبارح بس حاسس إنه مش مناسبني، أقدر أرجّعه خلال قد إيه؟",
                "تقدر ترجّع المنتج خلال 14 يوم من تاريخ الاستلام طالما ما اتستخدمش والتغليف الأصلي سليم ومعاه الفاتورة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن أرجّع منتج بعد أسبوع لو لسه مقفول بعلبته؟",
                "أيوه طبعاً، مسموح بالترجيع خلال 14 يوم من تاريخ الاستلام بشرط المنتج يكون غير مستخدم وفي تغليفه الأصلي ومعاه الفاتورة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "إيه شروط الاسترجاع العادي؟",
                "الاسترجاع متاح خلال 14 يوم من الاستلام بشرط إن المنتج ما يكونش مستخدم، والتغليف الأصلي يكون سليم، والفاتورة مرفقة معاه {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "رميت الفاتورة والعلبة بس ما استعملتش الجهاز، ينفع يرجع؟",
                "للأسف الاسترجاع بيشترط بقاء التغليف الأصلي سليم وإرفاق الفاتورة مع المنتج خلال مهلة الـ 14 يوم من الاستلام {c0}.",
            ),
            TurnVariant(
                "en",
                "What is your standard return window and what conditions apply?",
                "Unused items can be returned within 14 days of delivery as long as the original packaging remains intact and the invoice is included {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I return an unopened product 10 days after delivery?",
                "Yes, unused products in intact original packaging with the invoice can be returned within 14 days of delivery {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ret_defective_30d",
        split="train",
        domain="returns",
        behavior="grounded_single",
        supporting_chunks=(
            "في حال وجود عيب صناعة، يحق للعميل الإرجاع أو الاستبدال خلال 30 يوماً مع سحب مجاني للمنتج عبر المندوب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "الجهاز طلع فيه عيب صناعة بعد 20 يوم من الاستلام، لسه ينفع أرجّعه ولا الـ 14 يوم خلصت؟",
                "لو فيه عيب صناعة، ليك حق الإرجاع أو الاستبدال خلال 30 يوم مش 14، والمندوب بيسحب المنتج مجاناً {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لما المنتج يطلع بايظ من الأول، بدفع فلوس للمندوب اللي هييجي ياخده؟",
                "لأ يا فندم، في حالات عيوب الصناعة السحب بيكون مجاني عن طريق المندوب خلال مدة 30 يوم وتقدر تختار استرجاع أو استبدال {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "في عيب تصنيع في السماعة، ينفع أبدّلها بواحدة جديدة؟",
                "أيوه، لو في عيب صناعة يحق لك الاستبدال أو الإرجاع خلال 30 يوماً مع سحب مجاني عبر المندوب {c0}.",
            ),
            TurnVariant(
                "ar_to_en",
                "My product has a manufacturing defect after 3 weeks. Can I still replace it?",
                "Yes, items with a manufacturing defect qualify for return or replacement within 30 days with free courier pickup {c0}.",
            ),
            TurnVariant(
                "ar_to_en",
                "Do I pay pickup fees for a defective item return?",
                "No, courier pickup is completely free for manufacturing defect returns or replacements within 30 days {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ret_non_returnable",
        split="train",
        domain="returns",
        behavior="grounded_single",
        supporting_chunks=(
            "منتجات العناية الشخصية المختومة وسماعات الأذن المفتوحة والأكواد الرقمية المفعّلة لا تقبل الاسترجاع إلا إذا ثبت وجود عيب صناعة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "فتحت سماعة الأذن وجربتها بس صوتها مش عاجبني، ينفع أرجّعها؟",
                "للأسف سماعات الأذن المفتوحة غير قابلة للاسترجاع إلا لو ثبت إن فيها عيب صناعة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "اشتريت كارت شحن رقمي وفعلته على حسابي وعاوز أرجعه، ممكن؟",
                "الأكواد الرقمية المفعّلة مش بتتقبل في الاسترجاع إلا لو فيها عيب صناعة مثبت {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "إيه الحاجات اللي ما ينفعش ترجع بعد ما تتفتح؟",
                "منتجات العناية الشخصية المختومة وسماعات الأذن المفتوحة والأكواد الرقمية المفعّلة غير قابلة للاسترجاع إلا في حالة وجود عيب صناعة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "فتحت ماكينة حلاقة شخصية وبفكر أرجعها، ينفع؟",
                "منتجات العناية الشخصية المختومة بعد فتحها لا تقبل الاسترجاع إلا لو ثبت وجود عيب صناعة {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I return opened in-ear headphones because I didn't like the fit?",
                "Opened earphones, sealed personal-care products, and activated digital codes are non-returnable unless a manufacturing defect is verified {c0}.",
            ),
            TurnVariant(
                "en",
                "Are redeemed digital voucher codes eligible for return?",
                "No, activated digital codes cannot be returned unless proven defective {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ret_refund_timeline",
        split="train",
        domain="returns",
        behavior="grounded_multi",
        supporting_chunks=(
            "يستغرق فحص المرتجع في المستودع 48 ساعة عمل من وقت استلامه من المندوب.",
            "بعد اعتماد المرتجع، يُرد المبلغ إلى البطاقة البنكية أو InstaPay خلال 5 إلى 7 أيام عمل، بينما يُضاف رصيد المحفظة فوراً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "المندوب خد المرتجع مني إمبارح وأنا دافع بـ Visa، الفلوس هترجع إمتى؟",
                "أولاً المنتج بيتفحص في المستودع خلال 48 ساعة عمل من استلام المندوب {c0}، وبعد اعتماد المرتجع الفلوس بترجع على البطاقة البنكية خلال 5 إلى 7 أيام عمل {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "إيه أسرع طريقة أسترد بيها فلوس المرتجع، المحفظة ولا InstaPay؟",
                "بعد فحص المرتجع خلال 48 ساعة عمل {c0}، رصيد المحفظة بينزل فوراً بعد الاعتماد، إنما InstaPay والبطاقة البنكية بياخدوا من 5 إلى 7 أيام عمل {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "دفعت بـ InstaPay ورجّعت المنتج، ليه الفلوس لسه موصلتش بعد يومين؟",
                "الفحص في المستودع بياخد 48 ساعة عمل بعد سحب المندوب {c0}، وبعد الاعتماد التحويل على InstaPay بياخد من 5 لـ 7 أيام عمل {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو اخترت الفلوس تنزلي على محفظة المتجر بتاخد كام يوم بعد الفحص؟",
                "الفحص بياخد 48 ساعة عمل {c0}، وأول ما يتعتمد المرتجع الرصيد بينزل في محفظتك فوراً {c1}.",
            ),
            TurnVariant(
                "en",
                "How long does the whole refund process take for a credit card order after courier pickup?",
                "Warehouse inspection takes 48 business hours from pickup {c0}, and once approved, card or InstaPay refunds arrive within 5 to 7 business days {c1}.",
            ),
            TurnVariant(
                "en",
                "Is store wallet refund faster than InstaPay after return approval?",
                "Yes. After the 48-business-hour warehouse inspection {c0}, store wallet credit is issued instantly upon approval, whereas InstaPay or cards take 5 to 7 business days {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ret_cod_refund",
        split="train",
        domain="returns",
        behavior="grounded_single",
        supporting_chunks=(
            "طلبات الدفع عند الاستلام يتم رد قيمتها عبر تحويل InstaPay أو حساب بنكي أو رصيد محفظة، ولا يسلّم المندوب مبالغ نقدية.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "أنا دفعت كاش للمندوب، لما أرجّع الأوردر المندوب هيديني فلوسي كاش وهو بياخد الكرتونة؟",
                "لأ يا فندم، المندوب ما بيسلمش مبالغ نقدية؛ قيمة طلبات الدفع عند الاستلام بترجعلك عن طريق تحويل InstaPay أو تحويل بنكي أو رصيد محفظة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "طرق استرداد الفلوس إيه لو كنت دافع عند الاستلام؟",
                "طلبات الدفع عند الاستلام بتترد عن طريق InstaPay أو حساب بنكي أو كرصيد في المحفظة، ومفيش تسليم كاش مع المندوب {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن ترجّعولي فلوس الكاش على InstaPay؟",
                "أيوه طبعاً، استرداد طلبات الدفع عند الاستلام متاح عبر تحويل InstaPay أو حساب بنكي أو رصيد محفظة {c0}.",
            ),
            TurnVariant(
                "ar_to_en",
                "I paid Cash on Delivery. Will the courier hand me cash when picking up my return?",
                "No, couriers do not disburse cash. COD order refunds are sent via InstaPay, bank transfer, or store wallet credit {c0}.",
            ),
            TurnVariant(
                "ar_to_en",
                "How are Cash on Delivery refunds processed?",
                "COD refunds are transferred via InstaPay, bank account transfer, or store wallet credit rather than cash through the courier {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ret_fee_responsibility",
        split="train",
        domain="returns",
        behavior="grounded_multi",
        supporting_chunks=(
            "عند الإرجاع بسبب تغيير الرأي أو عدم ملاءمة المقاس دون عيب، يُخصم مبلغ 50 جنيهاً مقابل سحب المندوب.",
            "الإرجاع الناتج عن وصول منتج خاطئ أو تالف لا يتحمل فيه العميل أي رسوم شحن أو سحب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عاوز أرجع جاكيت عشان لونه معجبنيش في الحقيقة ومفهوش أي عيب، هدفع مصاريف سحب؟",
                "في حالة الإرجاع لتغيير الرأي أو عدم المناسبة بدون عيب، بيتخصم 50 جنيه مقابل سحب المندوب {c0}؛ الإعفاء الكامل من الرسوم بيكون بس لو المنتج وصل غلط أو تالف {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "وصلتني شنطة مقطوعة في الطلب، هل هتخصموا 50 جنيه لو رجعتها؟",
                "لأ خالص، الإرجاع بسبب وصول منتج تالف أو غلط العميل مش بيتحمل فيه أي رسوم سحب أو شحن {c1}؛ رسوم الـ 50 جنيه بتتخصم بس في حالة تغيير الرأي أو عدم ملاءمة المقاس بدون عيب {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "بعتولي موديل تاني غير اللي طلبته، هدفع حاجة عشان أرجعه؟",
                "بما إن المنتج وصل غلط، حضرتك مش هتتحمل أي رسوم سحب أو شحن إطلاقاً {c1}.",
            ),
            TurnVariant(
                "en",
                "Is there a pickup deduction if I return a jacket just because I changed my mind?",
                "Yes, change-of-mind or non-defective size returns deduct a 50 EGP courier pickup fee {c0}, whereas wrong or damaged item returns carry zero fees {c1}.",
            ),
            TurnVariant(
                "en",
                "I received the wrong color item. Will I be charged the 50 EGP return fee?",
                "No. Returns due to receiving a wrong or damaged item have zero shipping or pickup charges {c1}; the 50 EGP fee only applies to change-of-mind returns {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ret_partial_bundle",
        split="train",
        domain="returns",
        behavior="grounded_single",
        supporting_chunks=(
            "Bundle offers discounted as one package must be returned in full with every included item; partial bundle returns are not accepted.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "اشتريت عرض باندل مكون من ماوس وكيبورد، ينفع أرجّع الماوس لوحده وأسيب الكيبورد؟",
                "لأ يا فندم، عروض الباندل المخفضة كباقة واحدة لازم ترجع كاملة بكل محتوياتها ومش متاح ترجيع جزء واحد منها فقط {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "لازم أرجع كل قطع الباكدج لو هعمل استرجاع؟",
                "أيوه، عروض الباندل لازم ترجع كاملة بجميع القطع الموجودة فيها، والإرجاع الجزئي غير مقبول {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I return just one item from a discounted bundle package?",
                "No, discounted bundle packages must be returned in full with all included items; partial returns are not accepted {c0}.",
            ),
            TurnVariant(
                "en",
                "Do I have to send back the whole bundle if I request a return?",
                "Yes, all items in a discounted bundle offer must be returned together {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "لو أرجع الهدايا المجانية اللي جت جوه الباندل ولا أسيبها؟",
                "عروض الباندل كباقة واحدة لازم ترجع كاملة بكل القطع المضمنة فيها بدون تجزئة {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ret_exchange_size",
        split="train",
        domain="returns",
        behavior="grounded_single",
        supporting_chunks=(
            "استبدال مقاسات الملابس والأحذية متاح مجاناً لمرة واحدة فقط لكل طلب خلال 14 يوماً بشرط توفر المقاس البديل بالمخزون.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "طلبت كوتشي ومقاسه طلع ضيق، ينفع أبدله بمقاس أكبر وهل هدفع مصاريف؟",
                "تقدر تبدّل مقاسات الملابس والأحذية مجاناً لمرة واحدة للطلب خلال 14 يوم، بشرط المقاس اللي عاوزه يكون متاح في المخزون {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "بدلت تيشرت مقاس L بـ M ولسه مش مظبوط، ينفع أبدله تاني مجاناً؟",
                "الاستبدال المجاني للمقاسات متاح لمرة واحدة فقط لكل طلب خلال 14 يوماً وبشرط توفر المقاس بالمخزون {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "كم مرة أقدر أبدل مقاس بنطلون مجاناً؟",
                "ليك استبدال مقاس مجاني مرة واحدة بس لكل طلب خلال 14 يوم لو المقاس البديل متوفر في المخزون {c0}.",
            ),
            TurnVariant(
                "en",
                "How many free size exchanges are allowed for clothing and shoes?",
                "One free size exchange per order is available within 14 days, provided the replacement size is in stock {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I swap my shoe size for free within two weeks?",
                "Yes, one free size exchange is offered per order within 14 days as long as the requested size is available in stock {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ret_missing_accessories",
        split="train",
        domain="returns",
        behavior="grounded_single",
        supporting_chunks=(
            "نقص أي ملحق أساسي مثل الشاحن أو الكابل أو الكتيب أو التيكت من المرتجع يؤدي إلى رفض الطلب أو خصم 15% من قيمته.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عاوز أرجع راوتر بس ضيعت كابل الباور بتاعه، هيرجع عادي؟",
                "نقص أي ملحق أساسي زي الكابل أو الشاحن من المرتجع بيؤدي إما لرفض الاسترجاع أو خصم 15% من قيمة المنتج {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو التيكت بتاع البنطلون اتقطع أو ضاع، ممكن يرجع؟",
                "فقدان التيكت أو أي ملحق أساسي من المرتجع ممكن يسبب رفض الاسترجاع أو خصم 15% من قيمته {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "كم نسبة الخصم لو نسيت أحط الكتيب أو الشاحن مع الجهاز المرتجع؟",
                "نقص أي ملحق أساسي زي الشاحن أو الكتيب بيعرّض المرتجع للرفض أو خصم 15% من القيمة {c0}.",
            ),
            TurnVariant(
                "en",
                "What happens if a returned device is missing its original cable or tag?",
                "Missing essential accessories such as the charger, cable, manual, or tag results in rejection of the return or a 15% deduction from its value {c0}.",
            ),
            TurnVariant(
                "en",
                "Is there a restocking fee if the manual is missing from my return?",
                "Yes, omitting any essential item like the manual, charger, cable, or tag can cause a 15% deduction or return rejection {c0}.",
            ),
        ),
    ),

    # -------------------- PAYMENTS & INSTALLMENTS --------------------
    ScenarioSpec(
        scenario_group="tr_pay_cod_limit",
        split="train",
        domain="payments",
        behavior="grounded_single",
        supporting_chunks=(
            "يتاح الدفع عند الاستلام للطلبات حتى 5000 جنيه مصري كحد أقصى؛ أما الطلبات الأعلى فتتطلب الدفع المسبق بالبطاقة أو المحفظة أو InstaPay.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "الأوردر بتاعي بـ 6500 جنيه، ليه اختيار الدفع عند الاستلام مش ظاهر عندي؟",
                "الدفع عند الاستلام متاح بس للطلبات لحد 5000 جنيه كحد أقصى؛ وأي طلب فوق المبلغ ده بيحتاج دفع مسبق بالبطاقة أو المحفظة أو InstaPay {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "أقصى مبلغ أقدر أدفعه كاش للمندوب كام؟",
                "الحد الأقصى للدفع عند الاستلام هو 5000 جنيه مصري، والأعلى من كده بيتدفع أونلاين بالبطاقة أو المحفظة أو InstaPay {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "طلبت موبايل بـ 4900 جنيه، ينفع أدفعه عند الاستلام؟",
                "أيوه ينفع، لأن الدفع عند الاستلام متاح لأي طلب لحد 5000 جنيه {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "عندي لابتوب بـ 22 ألف جنيه ممكن أدفعه عند الباب؟",
                "للأسف لأ، الدفع عند الاستلام حده الأقصى 5000 جنيه، والطلبات الأكبر بتتطلب دفع مسبق بالبطاقة أو المحفظة أو InstaPay {c0}.",
            ),
            TurnVariant(
                "en",
                "Why is Cash on Delivery disabled for my 7,200 EGP order?",
                "Cash on Delivery is capped at 5,000 EGP maximum; orders above that require online prepayment via card, wallet, or InstaPay {c0}.",
            ),
            TurnVariant(
                "en",
                "What is the maximum cart value allowed for Cash on Delivery?",
                "The maximum order total eligible for Cash on Delivery is 5,000 EGP; higher totals require card, mobile wallet, or InstaPay prepayment {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_pay_bank_installments",
        split="train",
        domain="payments",
        behavior="grounded_multi",
        supporting_chunks=(
            "التقسيط بدون فوائد (0%) متاح لمدة 6 أو 12 شهراً لبطاقات ائتمان CIB والبنك الأهلي المصري وQNB على المشتريات من 1500 جنيه فأكثر.",
            "خطط التقسيط الأطول لمدة 18 أو 24 شهراً تخضع لرسوم إدارية بنسبة 3% تُخصم مرة واحدة عند التفعيل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عندي كارت ائتمان البنك الأهلي وعاوز أقسّط جهاز بـ 4000 جنيه على 12 شهر، فيه فوايد؟",
                "بطاقات ائتمان البنك الأهلي وCIB وQNB متاح لها تقسيط 0% فوائد لمدة 6 أو 12 شهر على المشتريات من 1500 جنيه فما فوق {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "طب لو هقسّط نفس المبلغ على 24 شهر ببطاقة CIB هدفع رسوم إضافية؟",
                "التقسيط بـ 0% بيكون على 6 أو 12 شهر بس {c0}، أما خطط الـ 18 والـ 24 شهر فبيتخصم عليها رسوم إدارية 3% مرة واحدة وقت التفعيل {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "أقل مبلغ للتقسيط البنكي بدون فوائد كام وأنهي بنوك مشمولة؟",
                "الحد الأدنى للتقسيط بدون فوائد 1500 جنيه على 6 أو 12 شهر، ومتاح لبطاقات ائتمان CIB والبنك الأهلي وQNB {c0}؛ ولو اخترت 18 أو 24 شهر بيكون فيه 3% رسوم إدارية {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "ينفع أقسّط على 18 شهر بدون أي مصاريف إدارية؟",
                "لأ يا فندم، الـ 0% متاح لخطط 6 و12 شهر فقط {c0}، أما 18 أو 24 شهر فبتخضع لرسوم إدارية 3% تُخصم مرة واحدة عند التفعيل {c1}.",
            ),
            TurnVariant(
                "en",
                "Which banks support 0% installments and is there a fee for 24-month plans?",
                "0% interest installments over 6 or 12 months are available on purchases of 1,500 EGP+ with CIB, NBE, and QNB credit cards {c0}, while 18- and 24-month plans carry a one-time 3% admin fee {c1}.",
            ),
            TurnVariant(
                "en",
                "Can I split a 2,000 EGP QNB purchase over 12 months at 0%?",
                "Yes, QNB credit cards qualify for 0% installments over 6 or 12 months for orders of 1,500 EGP or more {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_pay_valu_checkout",
        split="train",
        domain="payments",
        behavior="grounded_single",
        supporting_chunks=(
            "الدفع عبر ValU يتطلب حساباً مفعّلاً وحد أدنى للسلة 600 جنيه، ويتم تأكيد المعاملة برمز OTP يرسل إلى رقم الهاتف المسجل لدى ValU.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "أقل سلة ينفع أشتريها بـ ValU كام وهل ليها كود تأكيد؟",
                "الشراء عبر ValU محتاج حساب مفعّل وحد أدنى للسلة 600 جنيه، والتأكيد بيتم بكود OTP بيوصل على الرقم المسجل في ValU {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "كود تأكيد ValU مجاش على رقمي الجديد اللي حطيته في الأوردر، ليه؟",
                "رمز الـ OTP الخاص بـ ValU بيتبعت لرقم الموبايل المسجل لدى حسابك في ValU نفسه، وكمان السلة لازم تكون على الأقل 600 جنيه {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "ينفع أشتري منتج بـ 400 جنيه عن طريق ValU؟",
                "لأ يا فندم، الحد الأدنى للسلة للدفع بـ ValU هو 600 جنيه وحساب ValU مفعّل {c0}.",
            ),
            TurnVariant(
                "en",
                "What are the requirements for paying with ValU at checkout?",
                "ValU checkout requires an active ValU account, a minimum cart of 600 EGP, and confirmation via an OTP sent to your ValU-registered phone number {c0}.",
            ),
            TurnVariant(
                "en",
                "Why did my ValU checkout fail on a 500 EGP order?",
                "ValU requires a minimum cart total of 600 EGP along with an active ValU account {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_pay_split_payment",
        split="train",
        domain="payments",
        behavior="grounded_single",
        supporting_chunks=(
            "يمكن دمج رصيد محفظة المتجر مع البطاقة البنكية أو الدفع عند الاستلام لسداد باقي قيمة الطلب في نفس الفاتورة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عندي 300 جنيه في محفظتي على الموقع والأوردر بـ 1000، ينفع أدفع الباقي كاش أو فيزا؟",
                "أيوه ينفع تدمج رصيد محفظة المتجر مع البطاقة البنكية أو الدفع عند الاستلام عشان تكمل باقي المبلغ في نفس الفاتورة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل مسموح الدفع بجزء من المحفظة والباقي عند الاستلام؟",
                "طبعاً، تقدر تستخدم رصيد محفظة المتجر وتكمل الباقي بالبطاقة أو عند الاستلام في نفس الفاتورة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو رصيد المحفظة ما كفاش الأوردر كله هضيعه؟",
                "لأ، تقدر تدمج رصيد المحفظة مع البطاقة البنكية أو الدفع عند الاستلام لسداد الباقي {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I pay part of my order with store wallet credit and the rest with my card?",
                "Yes, store wallet credit can be combined with either a bank card or Cash on Delivery to cover the remaining balance on the same order {c0}.",
            ),
            TurnVariant(
                "en",
                "Does your checkout allow splitting wallet balance with COD?",
                "Yes, you can combine store wallet balance with Cash on Delivery or a card on the same invoice {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_pay_corporate_po",
        split="train",
        domain="payments",
        behavior="grounded_multi",
        supporting_chunks=(
            "الشركات المسجلة ضريبياً يمكنها سداد الطلبات التي تتجاوز 25000 جنيه عبر تحويل بنكي لحساب الشركة الرسمي.",
            "يتم شحن طلبات التحويل البنكي بعد مطابقة الإيصال ومقاصة الخزينة خلال يوم إلى يومي عمل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "إحنا شركة وبنعمل طلب توريد بـ 40 ألف جنيه، ينفع ندفع بتحويل بنكي وهيتشحن إمتى؟",
                "الشركات المسجلة ضريبياً تقدر تدفع الطلبات اللي فوق 25000 جنيه بتحويل بنكي لحساب الشركة الرسمي {c0}، والشحن بيتم بعد مطابقة الإيصال ومقاصة الخزينة خلال يوم ليومي عمل {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "بعتنا التحويل البنكي النهارده، ليه حالة الطلب لسه معلقة؟",
                "طلبات التحويل البنكي بتتشحن بعد مطابقة الإيصال ومقاصة الخزينة وده بياخد من يوم ليومي عمل {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل التحويل البنكي للشركات متاح لأي مبلغ؟",
                "سداد الشركات المسجلة ضريبياً بالتحويل البنكي متاح للطلبات الأكبر من 25000 جنيه {c0}، وبيتشحن الطلب بعد مقاصة الخزينة خلال يوم إلى يومي عمل {c1}.",
            ),
            TurnVariant(
                "en",
                "Can a tax-registered company pay a 35,000 EGP order by bank wire, and when does it ship?",
                "Tax-registered companies can pay orders exceeding 25,000 EGP via official bank wire {c0}; dispatch occurs after receipt reconciliation and treasury clearance within 1 to 2 business days {c1}.",
            ),
            TurnVariant(
                "en",
                "How long does treasury clearance take for corporate wire payments?",
                "Treasury reconciliation and clearance for bank wire orders take 1 to 2 business days before dispatch {c1}.",
            ),
        ),
    ),

    # -------------------- WARRANTY & REPAIRS --------------------
    ScenarioSpec(
        scenario_group="tr_war_electronics_12m",
        split="train",
        domain="warranty",
        behavior="grounded_single",
        supporting_chunks=(
            "تشمل الأجهزة الإلكترونية ضمان وكيل محلي لمدة 12 شهراً ضد عيوب الصناعة فقط، ولا يغطي سوء الاستخدام أو الكسر أو السوائل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "الضمان بتاع الإلكترونيات عندكم مدته قد إيه وبيغطي لو الشاشة اتكسرت مني؟",
                "الأجهزة الإلكترونية عليها ضمان وكيل محلي لمدة 12 شهر لعيوب الصناعة بس، ومش بيغطي الكسر أو السوائل أو سوء الاستخدام {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "وقع شاي على الكيبورد بعد شهرين من الشراء، الضمان هيصلحه؟",
                "للأسف ضمان الـ 12 شهر خاص بعيوب الصناعة فقط ومش بيغطي حوادث السوائل أو سوء الاستخدام أو الكسر {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل الأجهزة الإلكترونية عليها ضمان محلي؟",
                "أيوه، عليها ضمان وكيل محلي لمدة 12 شهراً ضد عيوب الصناعة فقط {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "إيه الحالات اللي خارج الضمان؟",
                "ضمان الـ 12 شهر بيغطي عيوب الصناعة بس، ومش بيشمل الكسر أو السوائل أو سوء الاستخدام {c0}.",
            ),
            TurnVariant(
                "en",
                "What does the standard electronics warranty cover and exclude?",
                "Electronics include a 12-month local agent warranty covering manufacturing defects only; misuse, physical breakage, and liquid damage are excluded {c0}.",
            ),
            TurnVariant(
                "en",
                "Does the 12-month warranty cover accidental water spills?",
                "No, the 12-month local agent warranty covers manufacturing defects only and excludes liquid damage, breakage, and misuse {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_war_battery_accessories",
        split="train",
        domain="warranty",
        behavior="grounded_multi",
        supporting_chunks=(
            "بطاريات اللابتوب والهواتف والشواحن المرفقة داخل العلبة مضمونة لمدة 6 أشهر فقط ضد عيوب التصنيع.",
            "الهدايا المجانية مثل الجرابات والكابلات الترويجية لا تخضع لأي ضمان بعد انقضاء فترة الـ 14 يوماً الأولى.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "بطارية اللابتوب باظت بعد 5 شهور، هي داخلة في ضمان الـ 12 شهر ولا مدتها مختلفة؟",
                "بطاريات اللابتوب والموبايل والشواحن اللي جوه العلبة ضمانها 6 شهور بس ضد عيوب التصنيع، فبعد 5 شهور لسه في الضمان {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "الكابل الهدية اللي جه مع الباوربانك باظ بعد شهرين، يتغير في الضمان؟",
                "للأسف الهدايا المجانية زي الجرابات والكابلات الترويجية ملهاش ضمان بعد أول 14 يوم {c1}؛ ضمان الـ 6 شهور بيكون للبطاريات والشواحن الأساسية اللي داخل العلبة بس {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "الشاحن الأصلي اللي في علبة الموبايل عليه ضمان قد إيه؟",
                "الشواحن المرفقة داخل العلبة والبطاريات عليها ضمان 6 شهور فقط ضد عيوب التصنيع {c0}.",
            ),
            TurnVariant(
                "en",
                "How long are inbox chargers, batteries, and free promotional cables covered?",
                "Phone and laptop batteries plus inbox chargers carry a 6-month manufacturing defect warranty {c0}, while free promotional cables and cases have no warranty after the first 14 days {c1}.",
            ),
            TurnVariant(
                "en",
                "My free bonus case tore after 3 weeks. Is it under warranty?",
                "No, free gifts like cases and promotional cables have no warranty coverage after the initial 14-day window {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_war_service_turnaround",
        split="train",
        domain="warranty",
        behavior="grounded_multi",
        supporting_chunks=(
            "يستغرق الفحص والإصلاح في مركز الخدمة المعتمد مدة تصل إلى 10 أيام عمل من تاريخ الاستلام.",
            "إذا تجاوزت مدة الصيانة 14 يوم عمل دون تسليم الجهاز، يحصل العميل على جهاز بديل أو رصيد استبدال كامل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "سلمت التابلت للضمان بقالي 6 أيام عمل، المدة المعتادة قد إيه؟ ولو اتأخروا عن أسبوعين إيه حقي؟",
                "الفحص والصيانة في المركز المعتمد بياخدوا لحد 10 أيام عمل {c0}. ولو الصيانة عدت 14 يوم عمل من غير استلام، من حقك تاخد جهاز بديل أو رصيد استبدال كامل {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "الجهاز في الصيانة بقالي 16 يوم عمل ولسه مخلصش، أعمل إيه؟",
                "طالما الصيانة تجاوزت 14 يوم عمل بدون ما تستلم الجهاز، يحق لحضرتك الحصول على جهاز بديل أو رصيد استبدال كامل {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "أقصى مدة طبيعية لتصليح الضمان كام يوم؟",
                "الفحص والإصلاح في مركز الخدمة المعتمد بياخد لحد 10 أيام عمل {c0}، ولو تخطى 14 يوم عمل بتاخد بديل أو رصيد استبدال {c1}.",
            ),
            TurnVariant(
                "en",
                "How long do warranty repairs take and what happens if they take over two weeks?",
                "Inspection and repair at the authorized center take up to 10 business days {c0}. If repairs exceed 14 business days without delivery, you receive a replacement unit or full store credit {c1}.",
            ),
            TurnVariant(
                "en",
                "My device has been in authorized repair for 15 business days. What are my options?",
                "Because the repair exceeded 14 business days, you are entitled to a replacement device or full replacement credit {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_war_dead_pixel",
        split="train",
        domain="warranty",
        behavior="grounded_single",
        supporting_chunks=(
            "يُعتمد استبدال شاشات الكمبيوتر بالضمان خلال أول 14 يوماً عند وجود 3 بيكسلات ميتة فأكثر أو بيكسل مضيء واحد في المنطقة الوسطى.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "شاشة الكمبيوتر لسه جايالي امبارح وفيها بيكسل واحد ميت في الطرف، هتتبدل بالضمان؟",
                "الاستبدال خلال أول 14 يوم بيتعتمد عند وجود 3 بيكسلات ميتة أو أكتر، أو بيكسل واحد مضيء في المنطقة الوسطى من الشاشة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "عندي نقطة منورة ثابتة في نص الشاشة الجديدة، ينفع أبدلها؟",
                "أيوه، وجود بيكسل مضيء واحد في المنطقة الوسطى خلال أول 14 يوم بيستوجب الاستبدال بالضمان {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "سياسة البيكسل الميت في الشاشات عندكم إيه؟",
                "خلال أول 14 يوم، بيتبدل المنتج لو فيه 3 بيكسلات ميتة فأكثر أو بيكسل مضيء واحد في منتصف الشاشة {c0}.",
            ),
            TurnVariant(
                "en",
                "What is the dead pixel threshold for monitor warranty replacement?",
                "Within the first 14 days, monitor warranty replacement is approved for 3 or more dead pixels or 1 bright stuck sub-pixel in the center zone {c0}.",
            ),
            TurnVariant(
                "en",
                "My new monitor has 4 dead pixels on day 5. Can I get it replaced?",
                "Yes, having 3 or more dead pixels within the first 14 days qualifies for warranty replacement {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_war_pickup_free",
        split="train",
        domain="warranty",
        behavior="grounded_single",
        supporting_chunks=(
            "سحب أجهزة الضمان مجاني داخل القاهرة الكبرى والإسكندرية، بينما في المحافظات الأخرى يتم التسليم لأقرب فرع شريك أو الشحن مدفوعاً مسبقاً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عاوز أبعت جهاز في الضمان وأنا في القاهرة، هروح فين ولا المندوب بيجي ياخده؟",
                "سحب أجهزة الضمان في القاهرة الكبرى وإسكندرية بيكون مجاني عن طريق المندوب {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "أنا في بني سويف والجهاز محتاج صيانة ضمان، هتدفعوا مصاريف المندوب؟",
                "السحب المجاني لأجهزة الضمان متاح في القاهرة الكبرى وإسكندرية بس؛ في المحافظات التانية التسليم بيكون لأقرب فرع شريك أو عن طريق شحن مدفوع مسبقاً {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل فيه سحب مجاني للضمان في إسكندرية؟",
                "أيوه، السحب المجاني لأجهزة الضمان متاح داخل القاهرة الكبرى والإسكندرية {c0}.",
            ),
            TurnVariant(
                "en",
                "Is warranty courier pickup free outside Cairo and Alexandria?",
                "Free warranty pickup covers Greater Cairo and Alexandria only; in other governorates, devices are dropped at the nearest partner branch or shipped prepaid {c0}.",
            ),
            TurnVariant(
                "en",
                "How do I send a defective device for warranty repair from Cairo?",
                "Within Greater Cairo and Alexandria, warranty device pickup by courier is completely free {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_war_smartwatch_water",
        split="train",
        domain="warranty",
        behavior="grounded_multi",
        supporting_chunks=(
            "الساعات الذكية المصنّفة بـ 5 ATM تتحمل المطر ورذاذ الماء والسباحة السطحية في حمامات السباحة.",
            "الغوص في المياه المالحة والتعرض لبخار الساونا الساخن يلغيان تغطية أضرار المياه في الضمان.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "الساعة 5 ATM دخلت فيها ميه وأنا في الساونا السخنة، الضمان هيغطيها؟",
                "للأسف لأ، رغم إن تصنيف 5 ATM بيتحمل المطر والسباحة السطحية في البسين {c0}، إلا إن التعرض لبخار الساونا الساخن أو الغوص في المياه المالحة بيلغي تغطية المياه في الضمان {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "ينفع ألبس الساعة الـ 5 ATM في البسين ولا هتبوظ الضمان؟",
                "الساعات المصنفة 5 ATM بتتحمل المطر والسباحة السطحية في حمامات السباحة {c0}، بس ممنوع الغوص في البحر أو الساونا السخنة عشان الضمان ما يتلغيش {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "نزلت البحر المالح بالساعة وعملت غوص والضمان رفضها، ليه؟",
                "لأن الغوص في المياه المالحة (وكذلك بخار الساونا الساخن) بيلغي تغطية أضرار المياه في الضمان {c1}؛ الحماية بتكون للمطر والسباحة السطحية فقط {c0}.",
            ),
            TurnVariant(
                "en",
                "Does 5 ATM smartwatch warranty cover hot steam saunas and saltwater diving?",
                "No. While 5 ATM watches handle rain and shallow pool swimming {c0}, saltwater diving and hot sauna steam void water-damage warranty coverage {c1}.",
            ),
            TurnVariant(
                "en",
                "Can I swim in a pool with my 5 ATM watch without voiding warranty?",
                "Yes, 5 ATM covers rain, splashes, and surface pool swimming {c0}, provided you avoid saltwater diving and hot steam saunas {c1}.",
            ),
        ),
    ),

    # -------------------- ORDERS & CANCELLATION --------------------
    ScenarioSpec(
        scenario_group="tr_ord_cancel_before_ship",
        split="train",
        domain="orders",
        behavior="grounded_single",
        supporting_chunks=(
            "يمكن إلغاء الطلب ذاتياً من صفحة طلباتي مجاناً قبل تحول حالته إلى 'خارج للتوصيل'؛ وبعد خروجه مع المندوب يكفي رفض الاستلام عند الباب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عاوز ألغي الأوردر قبل ما يتشحن، ينفع من حسابي؟",
                "أيوه، تقدر تلغي الطلب بنفسك مجاناً من صفحة طلباتي طول ما حالته لسه ما تحولتش إلى 'خارج للتوصيل' {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "الطلب بقى خارج للتوصيل ومش لاقي زرار الإلغاء، أعمل إيه؟",
                "بعد ما الطلب يخرج مع المندوب مش هيتلغي من الصفحة، بس يكفي ترفض استلامه عند الباب لما المندوب يوصلك {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل فيه غرامة لو لغيت الطلب وهو لسه بيتجهز؟",
                "لأ خالص، الإلغاء مجاني ذاتياً من صفحة طلباتي قبل ما الحالة تتحول لـ 'خارج للتوصيل' {c0}.",
            ),
            TurnVariant(
                "en",
                "How do I cancel an order that is already Out for Delivery?",
                "Self-service cancellation closes once the order moves to 'Out for Delivery'; after dispatch, simply decline the parcel at your door {c0}.",
            ),
            TurnVariant(
                "en",
                "Is there any fee to cancel from My Orders before dispatch?",
                "No, cancelling from your 'My Orders' page before the status becomes 'Out for Delivery' is completely free {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ord_open_box_policy",
        split="train",
        domain="orders",
        behavior="grounded_single",
        supporting_chunks=(
            "يُسمح للعميل بفحص التغليف الخارجي ومطابقة اسم الصنف وعدد القطع على الفاتورة قبل التوقيع، لكن تشغيل الأجهزة الإلكترونية المختومة يتطلب سداد القيمة أولاً مع بقاء ضمان 30 يوماً لعيوب الصناعة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "ينفع أفتح علبة الموبايل وأشغله أجربه قدام المندوب قبل ما أدفع الفلوس؟",
                "مسموح تفحص التغليف الخارجي وتطابق الصنف والعدد مع الفاتورة قبل التوقيع، لكن فتح وتشغيل الأجهزة الإلكترونية المختومة بيتطلب الدفع الأول وبيفضل معاك ضمان 30 يوم لعيوب الصناعة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "إيه اللي مسموحلي أتأكد منه قبل ما أمضي للمندوب؟",
                "تقدر تفحص التغليف الخارجي وتتأكد من اسم الصنف وعدد القطع في الفاتورة قبل التوقيع {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو دفعت وفتحت الجهاز بعدها لقيت فيه عيب، حقي بيضيع؟",
                "لأ أبداً، حتى بعد سداد القيمة وفتح الجهاز المختوم بيفضل عندك ضمان 30 يوماً ضد عيوب الصناعة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "المندوب رفض يخليني أفتح برشام اللابتوب قبل ما أستلم، هل ده الصح؟",
                "أيوه ده الإجراء الصح: المندوب يسمح بفحص الكرتونة الخارجية ومطابقة الفاتورة قبل التوقيع، أما تشغيل الجهاز المختوم فبيكون بعد الدفع مع حفظ حقك في ضمان 30 يوم لعيوب الصناعة {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I unseal and power on an electronic device in front of the courier before paying?",
                "You may inspect the outer seal and verify item names and counts against the invoice before signing, while unsealing and powering on sealed electronics requires payment first (30-day defect protection still applies) {c0}.",
            ),
            TurnVariant(
                "en",
                "What am I allowed to check at delivery before signing for a sealed box?",
                "Before signing, you can inspect the outer packaging and match the item name and quantity with the invoice; powering on sealed electronics happens after payment {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ord_split_shipment",
        split="train",
        domain="orders",
        behavior="grounded_multi",
        supporting_chunks=(
            "الطلبات التي تضم منتجات من أكثر من بائع قد تصل في طرود منفصلة برقم تتبع مستقل لكل طرد.",
            "يُحتسب رسم الشحن لمرة واحدة فقط عند إتمام الدفع ولا يتكرر بتعدد الطرود.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "وصلني طرد واحد وفيه منتج ناقص مع إني طلبت من بائعين في أوردر واحد، وهدفع شحن تاني لما التاني ييجي؟",
                "لما الطلب يكون من أكتر من بائع ممكن يوصل على طرود منفصلة ولكل طرد رقم تتبع لوحده {c0}، ومش هتدفع شحن تاني لأن رسوم الشحن بتتحسب مرة واحدة بس وقت الدفع {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "ليه الأوردر بتاعي اتقسم شحنتين وبرقمين تتبع؟",
                "عشان لما تشتري منتجات من بائعين مختلفين، كل جزء ممكن يوصل في طرد منفصل برقم تتبع خاص بيه، لكن رسم الشحن بيفضل مرة واحدة بس {c0}{c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "المندوب جابلي نص الطلب، هل هيطلب شحن زيادة على النص التاني؟",
                "لأ، رسوم الشحن بتتحسب مرة واحدة بس وقت إتمام الدفع ومش بتتكرر مهما اتعددت الطرود {c1}.",
            ),
            TurnVariant(
                "en",
                "My order was split into two tracking numbers. Will I pay double shipping?",
                "No. Multi-seller carts may arrive in separate packages with their own tracking IDs {c0}, but shipping is charged only once at checkout regardless of package count {c1}.",
            ),
            TurnVariant(
                "en",
                "Why did only half of my multi-vendor order arrive today?",
                "Items from multiple sellers may ship in separate packages with independent tracking numbers {c0}, without any extra shipping charge {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ord_guest_tracking",
        split="train",
        domain="orders",
        behavior="grounded_single",
        supporting_chunks=(
            "يمكن لعملاء الشراء كزائر (Guest) متابعة شحناتهم عبر إدخال رقم الطلب ورقم الهاتف المستخدم في الدفع بدون إنشاء حساب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "اشتريت من غير ما أعمل حساب (Guest)، أتابع شحنتي إزاي؟",
                "تقدر تتابع طلبك بسهولة من غير حساب عن طريق إدخال رقم الطلب ورقم الموبايل اللي سجلت بيه وقت الدفع {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لازم أعمل حساب عشان أشوف تتبع الشحنة؟",
                "لأ مش لازم، تقدر تتابع طلب الزائر برقم الطلب ورقم التليفون المستخدم في الدفع {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "محتاج إيه عشان أفتح صفحة تتبع أوردر الـ Guest؟",
                "محتاج فقط رقم الطلب ورقم الهاتف اللي استخدمته وقت إتمام الدفع {c0}.",
            ),
            TurnVariant(
                "en",
                "How do I track an order placed through Guest Checkout?",
                "Guest shoppers can track shipments without an account by entering the order ID and the phone number used at checkout {c0}.",
            ),
            TurnVariant(
                "en",
                "Do I need to register an account just to view guest order status?",
                "No registration is required; simply enter your order number and checkout phone number {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ord_gift_receipt",
        split="train",
        domain="orders",
        behavior="grounded_multi",
        supporting_chunks=(
            "تفعيل خيار 'طلب هدية' يخفي الأسعار من الفاتورة الورقية المطبوعة داخل الطرد.",
            "يستطيع مستلم الهدية طلب إرجاعها للحصول على رصيد متجر عبر إدخال رمز الهدية الموجود على القسيمة دون علم المشتري.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "ببعت أوردر هدية لصاحبي، هل السعر هيبان على الفاتورة؟ ولو مقاسه ما طلعش مظبوط يقدر يرجّعه؟",
                "لما تفعّل خيار 'طلب هدية' السعر بيختفي من الفاتورة الورقية اللي جوه الطرد {c0}، ولو صاحبك حابب يرجّعها يقدر يستخدم رمز الهدية اللي على القسيمة وياخد رصيد متجر بدون ما توصلك رسالة {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو جتلي هدية من عندكم وعاوز أرجعها من غير ما أحرج اللي بعتها، الفلوس بترجعلي إزاي؟",
                "تقدر ترجّع الهدية باستخدام رمز الهدية الموجود على القسيمة عشان تاخد رصيد متجر من غير إخطار المشتري {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "إزاي أبعت هدية من غير ما السعر يظهر؟",
                "فعّل خيار 'طلب هدية' وقت الطلب، وده بيخفي الأسعار تماماً من الفاتورة الورقية جوه الطرد {c0}.",
            ),
            TurnVariant(
                "en",
                "If I mark an order as a gift, are prices hidden and can the recipient return it?",
                "Yes, selecting 'Gift Order' hides prices on the printed packing slip {c0}, and the recipient can return it for store credit using the gift code on the slip without notifying the buyer {c1}.",
            ),
            TurnVariant(
                "en",
                "How does a gift recipient get store credit for an unwanted gift?",
                "The recipient can enter the gift code printed on the gift slip to return the item for store credit privately {c1}.",
            ),
        ),
    ),

    # -------------------- MARKETPLACE LISTINGS & SELLERS --------------------
    ScenarioSpec(
        scenario_group="tr_lst_verified_badge",
        split="train",
        domain="listings",
        behavior="grounded_single",
        supporting_chunks=(
            "شارة 'بائع موثق' تتطلب سجل تجاري ساري وبطاقة ضريبية مصرية ومعدل إلغاء من البائع أقل من 2% خلال آخر 60 يوماً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "إيه شروط الحصول على علامة بائع موثق (Verified Seller)؟",
                "عشان تاخد شارة 'بائع موثق' لازم تقدم سجل تجاري ساري وبطاقة ضريبية مصرية، ويكون معدل إلغاء الطلبات من طرفك أقل من 2% في آخر 60 يوم {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "معدل الإلغاء المسموح لبائع موثق كام؟",
                "لازم معدل الإلغاء من البائع يكون أقل من 2% خلال آخر 60 يوم بجانب سجل تجاري ساري وبطاقة ضريبية مصرية {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "ينفع أخد شارة بائع موثق من غير بطاقة ضريبية؟",
                "لأ يا فندم، البطاقة الضريبية المصرية والسجل التجاري الساري شرط أساسي مع معدل إلغاء أقل من 2% خلال 60 يوم {c0}.",
            ),
            TurnVariant(
                "en",
                "What are the requirements for a Verified Seller badge?",
                "The Verified Seller badge requires a valid commercial register, an Egyptian tax card, and a seller-initiated cancellation rate below 2% over the last 60 days {c0}.",
            ),
            TurnVariant(
                "en",
                "Does a verified seller need a tax card?",
                "Yes, both an Egyptian tax card and a valid commercial register are required, alongside keeping cancellations under 2% in the past 60 days {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_lst_condition_grades",
        split="train",
        domain="listings",
        behavior="grounded_multi",
        supporting_chunks=(
            "إعلانات الأجهزة المجددة فئة Refurbished Grade A تشمل ضمان بائع لمدة 6 أشهر وبحد أدنى 85% لصحة البطارية.",
            "إعلانات السلع المستعملة (Used) يجب أن تتضمن 3 صور حقيقية للمنتج ووصفاً صريحاً لأي خدوش أو عيوب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "إيه الفرق بين إعلان Refurbished Grade A وإعلان Used في الضمان والبطارية؟",
                "الأجهزة فئة Refurbished Grade A بيجي معاها ضمان بائع 6 شهور وصحة بطارية لا تقل عن 85% {c0}، أما الإعلانات المستعملة Used فشرطها الأساسي وجود 3 صور حقيقية وتوضيح صريح لأي خدوش أو عيوب {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "عاوز أعرض لابتوب مستعمل للبيع، أرفع كام صورة وإيه المطلوب في الوصف؟",
                "إعلانات السلع المستعملة بتتطلب رفع 3 صور حقيقية للمنتج مع وصف صريح لأي خدوش أو عيوب موجودة فيه {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "موبايل Refurbished Grade A لقيت بطاريته 79% لما استلمته، ده مطابق للمواصفات؟",
                "لأ، فئة Refurbished Grade A لازم تكون صحة البطارية فيها 85% على الأقل ومعاها ضمان بائع 6 شهور {c0}.",
            ),
            TurnVariant(
                "en",
                "What battery health and warranty do Refurbished Grade A phones require compared to Used listings?",
                "Refurbished Grade A items must have at least 85% battery health and a 6-month seller warranty {c0}, whereas Used listings require 3 real photos and explicit disclosure of any scratches or defects {c1}.",
            ),
            TurnVariant(
                "en",
                "How many photos are mandatory to publish a Used item listing?",
                "Used marketplace listings must include at least 3 real photos of the item along with an explicit description of any scratches or flaws {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_lst_seller_payout",
        split="train",
        domain="listings",
        behavior="grounded_single",
        supporting_chunks=(
            "تُحوّل مستحقات البائعين بعد 7 أيام من انتهاء مهلة الاسترجاع المقررة للمشتري مع خصم عمولة منصة موحدة قدرها 8%.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "أنا بائع وبعت أوردر واتسلّم للعميل، فلوسي هتحولولي إمتى وعمولتكم كام؟",
                "مستحقاتك بتتحول بعد 7 أيام من انتهاء مهلة استرجاع المشتري، وبتتخصم عمولة المنصة الموحدة 8% {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "نسبة عمولة المنصة من البائعين كام في المية؟",
                "عمولة المنصة الموحدة هي 8% وبتتخصم عند تحويل المستحقات بعد 7 أيام من نهاية مهلة الاسترجاع {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "ليه الفلوس ما نزلتش يوم التسليم للمشتري على طول؟",
                "المستحقات بتتحول بعد مرور 7 أيام من انتهاء مهلة استرجاع المشتري بعد خصم عمولة 8% {c0}.",
            ),
            TurnVariant(
                "en",
                "When do sellers get paid and what is the marketplace commission rate?",
                "Seller payouts are released 7 days after the buyer's return window closes, minus a flat 8% platform commission {c0}.",
            ),
            TurnVariant(
                "en",
                "What commission percentage is deducted from seller payouts?",
                "A unified 8% platform commission is deducted when seller payouts transfer 7 days post-return-window {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_lst_used_inspection",
        split="train",
        domain="listings",
        behavior="grounded_multi",
        supporting_chunks=(
            "الهواتف المستعملة التي تتجاوز قيمتها 8000 جنيه تمر بفحص معمل Sanad المكون من 20 نقطة قبل تسليمها للمشتري.",
            "يصدر تقرير فحص معملي صالح لمدة 48 ساعة ويحق للمشتري إلغاء الصفقة فوراً قبل التوصيل إذا ظهر عيب غير معلن.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "هشتري موبايل مستعمل بـ 12 ألف جنيه من بائع، إزاي أضمن إنه سليم قبل ما يوصلني؟",
                "أي موبايل مستعمل فوق 8000 جنيه بيعدي على فحص معمل Sanad المكون من 20 نقطة قبل التسليم {c0}، وبيطلعلك تقرير فحص صالح لمدة 48 ساعة ولو ظهر عيب مش معلن تقدر تلغي الصفقة فوراً قبل التوصيل {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "تقرير فحص المعمل لقيت فيه مشكلة البائع ما ذكرهاش، لازم أكمل الشراء؟",
                "لأ خالص، يحقلك تلغي الصفقة فوراً قبل التوصيل لو التقرير كشف أي عيب غير معلن {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "إيه الحد السعري اللي بيخلي الموبايل المستعمل يدخل فحص الـ 20 نقطة؟",
                "الهواتف المستعملة اللي سعرها يتعدى 8000 جنيه بتخضع لفحص معمل Sanad من 20 نقطة قبل التوصيل {c0}.",
            ),
            TurnVariant(
                "en",
                "How are used phones over 8,000 EGP inspected and what if an undisclosed fault is found?",
                "Used phones priced above 8,000 EGP go through Sanad's 20-point lab check before delivery {c0}; if the 48-hour valid report shows an undisclosed defect, the buyer can cancel immediately before delivery {c1}.",
            ),
            TurnVariant(
                "en",
                "How long is the Sanad 20-point lab inspection report valid?",
                "The lab inspection report remains valid for 48 hours and lets buyers cancel immediately if undisclosed faults are detected {c1}.",
            ),
        ),
    ),

    # -------------------- ACCOUNT, SECURITY & PROMOTIONS --------------------
    ScenarioSpec(
        scenario_group="tr_acc_merge_blocked",
        split="train",
        domain="account",
        behavior="grounded_single",
        supporting_chunks=(
            "لا يدعم النظام دمج حسابين مسجلين برقمي هاتف مختلفين؛ وتبقى النقاط وسجل الطلبات مرتبطة بالحساب الأصلي فقط.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عندي حسابين برقمين مختلفين، ينفع تدمجوهم وتنقلوا النقاط والطلبات في حساب واحد؟",
                "للأسف النظام ما بيدعمش دمج حسابين برقمين مختلفين، والنقاط وسجل الطلبات بتفضل مربوطة بالحساب الأصلي بس {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن أنقل سجل طلباتي من رقمي القديم لرقمي الجديد في حساب تاني؟",
                "لأ يا فندم، ما فيش إمكانية لدمج الحسابات أو نقل النقاط وسجل الطلبات بين رقمين مختلفين {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل ينفع توحيد نقط الولاء بين حسابين؟",
                "لأ، دمج حسابين برقمي موبايل مختلفين غير مدعوم، والنقاط بتفضل جوه الحساب الأصلي {c0}.",
            ),
            TurnVariant(
                "en",
                "Can you merge two accounts registered under different phone numbers?",
                "No, merging accounts across different phone numbers is not supported; loyalty points and order history remain tied to the original account {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I transfer my loyalty points to another account with a new number?",
                "No, points and order history cannot be transferred or merged between two separate phone-number accounts {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_acc_trusted_devices",
        split="train",
        domain="account",
        behavior="grounded_single",
        supporting_chunks=(
            "يمكنك إنهاء الجلسات النشطة من صفحة إعدادات الأمان، كما يؤدي تغيير كلمة المرور إلى تسجيل الخروج الإجباري من كافة الأجهزة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "موبايلي اتسرق وحسابي لسه مفتوح عليه، أخرّج الحساب منه إزاي؟",
                "تقدر تنهي أي جلسة نشطة من صفحة إعدادات الأمان، أو تغير كلمة المرور وده هيعمل تسجيل خروج إجباري من كل الأجهزة فوراً {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو غيرت الباسورد هل الحساب هيتقفل على اللابتوب والتليفون التاني؟",
                "أيوه، تغيير كلمة المرور بيجبر تسجيل الخروج من جميع الأجهزة المفتوحة، وكمان تقدر تدير الجلسات من إعدادات الأمان {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "فين أقدر أشوف وأقفل الأجهزة اللي فاتحة حسابي؟",
                "من صفحة إعدادات الأمان تقدر تنهي أي جلسات نشطة، أو تغير كلمة المرور لتسجيل الخروج من كل الأجهزة {c0}.",
            ),
            TurnVariant(
                "en",
                "How do I sign out my account remotely from a lost device?",
                "You can revoke active sessions from Security Settings, and resetting your password automatically forces logout across all devices {c0}.",
            ),
            TurnVariant(
                "en",
                "Does changing my password log out other logged-in browsers?",
                "Yes, changing your password triggers a forced sign-out on all active devices {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_promo_stacking",
        split="train",
        domain="promotions",
        behavior="grounded_multi",
        supporting_chunks=(
            "يُسمح باستخدام كود خصم واحد فقط لكل طلب ولا يمكن دمجه مع الخصومات الفورية لبطاقات البنوك.",
            "أكواد الخصم لا تنطبق على قسائم الهدايا الرقمية أو رسوم الشحن.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "ليه الكوبون اتلغى لما اخترت خصم بنك CIB الفوري، وكمان ما خصمش من مصاريف الشحن؟",
                "مسموح بكود خصم واحد بس في الطلب ومش بيتجمع مع الخصومات البنكية الفورية {c0}، وكمان أكواد الخصم مش بتشتغل على رسوم الشحن أو قسائم الهدايا {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "عندي كوبونين خصم، ينفع أحطهم هما الاتنين على نفس الأوردر؟",
                "لأ يا فندم، متاح كود خصم واحد بس لكل طلب ومش بيتدمج مع عروض البنوك الفورية {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "جربت أحط برومو كود على كارت هدية رقمي ورفض، ليه؟",
                "لأن أكواد الخصم لا تنطبق على قسائم الهدايا الرقمية أو مصاريف الشحن {c1}.",
            ),
            TurnVariant(
                "en",
                "Why won't my promo code apply alongside my bank card instant discount or cover shipping?",
                "Only one promo code is allowed per order and cannot stack with instant bank discounts {c0}; promo codes also exclude shipping fees and digital gift vouchers {c1}.",
            ),
            TurnVariant(
                "en",
                "Can I use a promo code to buy a digital gift card?",
                "No, promo codes do not apply to digital gift vouchers or shipping fees {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_promo_referral_bonus",
        split="train",
        domain="promotions",
        behavior="grounded_multi",
        supporting_chunks=(
            "يحصل الداعي على قسيمة 100 جنيه بمجرد تسليم أول طلب لصديقه بقيمة تزيد عن 500 جنيه.",
            "يتم تفعيل قسيمة الـ 100 جنيه في الحساب بعد مرور فترة الـ 14 يوماً الخاصة بالاسترجاع دون إلغاء.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "صاحبي سجل بالكود بتاعي واستلم طلب بـ 700 جنيه إمبارح، قسيمة الـ 100 جنيه منزلتليش ليه؟",
                "قسيمة الـ 100 جنيه بتستحقها على أول طلب لصديقك فوق 500 جنيه {c0}، لكنها بتتفعل في حسابك بعد ما تعدي فترة الـ 14 يوم بتاعة الاسترجاع من غير ما الطلب يتلغي {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "إمتى باخد مكافأة دعوة صديق ولو طلب بـ 300 جنيه هاخدها؟",
                "لازم أول طلب لصديقك يزيد عن 500 جنيه عشان تاخد قسيمة الـ 100 جنيه {c0}، وبتتفعل بعد مرور 14 يوماً من غير استرجاع {c1}.",
            ),
            TurnVariant(
                "en",
                "My friend completed their first 650 EGP order with my referral link yesterday. When do I get my 100 EGP voucher?",
                "Orders over 500 EGP qualify for the 100 EGP referral voucher {c0}, and it activates in your account once the 14-day return window passes without cancellation {c1}.",
            ),
            TurnVariant(
                "en",
                "What is the minimum order amount for a friend referral reward?",
                "Your friend's first delivered order must exceed 500 EGP, and the 100 EGP voucher unlocks after the 14-day return period {c0}{c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "صاحبي اشترى بكام عشان ينزلي بونص الـ 100 جنيه؟",
                "لازم أول طلب ليه يتسلّم وتكون قيمته أكتر من 500 جنيه {c0}، وتستنى الـ 14 يوم بتوع مهلة الاسترجاع وبعدها القسيمة تتفعل {c1}.",
            ),
        ),
    ),

    # -------------------- CROSS-LINGUAL & CONTRAST GROUNDED --------------------
    ScenarioSpec(
        scenario_group="tr_xl_international_warranty_no",
        split="train",
        domain="warranty",
        behavior="grounded_single",
        supporting_chunks=(
            "Imported parallel-stock accessories carry a 3-month seller warranty only and are not covered by local brand service centers.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "لو اشتريت إكسسوار مستورد (Parallel-Stock)، الوكيل المحلي هيقبله في الصيانة؟",
                "لأ يا فندم، الإكسسوارات المستوردة (Parallel-Stock) عليها ضمان بائع 3 شهور بس ومش مشمولة في مراكز خدمة البراند المحلي {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "ضمان الإكسسوارات المستوردة قد إيه ومنين؟",
                "المنتجات الإكسسوار المستوردة بتيجي بضمان بائع لمدة 3 شهور فقط، ومش بتدخل مراكز الوكيل المحلي {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "هل ضمان البائع للإكسسوار المستورد سنة كاملة؟",
                "لأ، ضمان الإكسسوارات المستوردة (Parallel-Stock) هو 3 شهور عن طريق البائع فقط {c0}.",
            ),
            TurnVariant(
                "en",
                "Are parallel-stock imported accessories serviced by the local official brand center?",
                "No, imported parallel-stock accessories carry only a 3-month seller warranty and are not covered by local brand service centers {c0}.",
            ),
            TurnVariant(
                "en",
                "What warranty applies to parallel-stock accessories?",
                "They include a 3-month seller-only warranty rather than local official brand center coverage {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_xl_preorder_deposit",
        split="train",
        domain="orders",
        behavior="grounded_multi",
        supporting_chunks=(
            "Pre-order reservations require a 20% deposit that becomes non-refundable once stock is allocated with the supplier.",
            "The paid 20% deposit is automatically deducted from the final balance charged at dispatch.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "في الحجز المسبق (Pre-order)، بدفع مقدم كام وهل بيتخصم من سعر الجهاز لما يتشحن؟",
                "الحجز المسبق بيتطلب مقدم 20% (بيكون غير قابل للاسترداد بمجرد تخصيص الكمية عند المورد) {c0}، والمبلغ ده بيتخصم تلقائياً من الرصيد النهائي وقت الشحن {c1}.",
            ),
            TurnVariant(
                "en_to_ar",
                "لو خصصوا الكمية عند المورد خلاص وقررت ألغي الحجز المسبق، المقدم الـ 20% هيرجعلي؟",
                "بعد ما الكمية تتخصص مع المورد، مقدم الـ 20% بيكون غير قابل للاسترداد {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "هل هدفع السعر كامل وقت الشحن فوق المقدم الـ 20%؟",
                "لأ طبعاً، الـ 20% اللي اتدفعت في الحجز بتتخصم أوتوماتيك من المبلغ النهائي وقت الشحن {c1}.",
            ),
            TurnVariant(
                "en",
                "How does the 20% pre-order deposit work if I cancel after supplier allocation?",
                "Pre-orders require a 20% deposit that is non-refundable once stock is allocated with the supplier {c0}, and that deposit is automatically deducted from your final dispatch balance {c1}.",
            ),
            TurnVariant(
                "en",
                "Is the pre-order deposit subtracted from the final bill when shipped?",
                "Yes, the 20% pre-order deposit is automatically deducted from the final balance charged upon dispatch {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_xl_customs_local_only",
        split="train",
        domain="payments",
        behavior="grounded_single",
        supporting_chunks=(
            "All listed prices for in-stock catalog items already include Egyptian customs clearance with zero extra fees on delivery.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "السعر المكتوب على الموقع للمنتجات المتوفرة شامل الجمارك ولا هدفع جمارك عند الاستلام؟",
                "كل الأسعار المعروضة للمنتجات المتوفرة في المخزون شاملة التخليص الجمركي المصري بالكامل ومش هتدفع أي رسوم إضافية عند الاستلام {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "في مصاريف جمركية زيادة غير السعر اللي في الصفحة؟",
                "لأ، الأسعار المعلنة للسلع المتوفرة في الكتالوج شاملة الجمارك المصرية بدون أي مصاريف عند الاستلام {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "هل المندوب هيطلب مني رسوم تخليص جمركي للمنتج؟",
                "لأ يا فندم، المنتجات المتوفرة سعرها شامل التخليص الجمركي ومفيش أي رسوم إضافية وقت التسليم {c0}.",
            ),
            TurnVariant(
                "en",
                "Will I have to pay Egyptian customs fees on delivery for stocked items?",
                "No, all prices displayed for in-stock catalog items already include Egyptian customs clearance with no extra fees at delivery {c0}.",
            ),
            TurnVariant(
                "en",
                "Are customs duties included in the product price?",
                "Yes, Egyptian customs clearance is fully included in listed prices for all in-stock items {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_xl_flash_sale_qty",
        split="train",
        domain="promotions",
        behavior="grounded_single",
        supporting_chunks=(
            "Flash sale deals are limited to 1 unit per account; duplicate orders for the same deal auto-cancel within 30 minutes.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "عملت طلبين لنفس عرض الـ Flash Sale والتاني اتلغى لوحده، ليه؟",
                "عروض الـ Flash Sale حدها قطعة واحدة بس لكل حساب، وأي طلب متكرر لنفس العرض بيتلغي تلقائياً خلال 30 دقيقة {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "مسموح بكام قطعة من عروض الفلاش سيل؟",
                "الحد الأقصى هو قطعة واحدة بس لكل حساب، ولو اتعمل طلب مكرر بيتلغي أوتوماتيك في خلال 30 دقيقة {c0}.",
            ),
            TurnVariant(
                "en",
                "Why did my second flash sale order cancel automatically after 20 minutes?",
                "Flash sale deals are capped at 1 unit per account, and duplicate orders automatically cancel within 30 minutes {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I buy 3 units of the same flash sale deal?",
                "No, flash sale items are restricted to 1 unit per account; duplicates auto-cancel within 30 minutes {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "أقدر أطلب حتتين من نفس عرض الـ Flash Sale في أوردرين منفصلين؟",
                "لأ، العرض محدد بقطعة واحدة لكل حساب والطلبات المكررة لنفس العرض بتتلغي تلقائياً خلال 30 دقيقة {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_contrast_no_price_match",
        split="train",
        domain="promotions",
        behavior="grounded_single",
        supporting_chunks=(
            "لا يوفر المتجر سياسة مطابقة أو تعويض فرق السعر بأثر رجعي بعد تسليم الطلب، حتى لو نزل المنتج في عرض أو خصم في اليوم التالي.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "استلمت المروحة إمبارح والنهارده لقيت عليها خصم 400 جنيه، ممكن تعوضوني بالفرق؟",
                "للأسف المتجر مش بيوفر تعويض أو مطابقة فرق السعر بأثر رجعي بعد تسليم الطلب، حتى لو السعر نزل في عرض تاني يوم {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو سعر المنتج نزل بعد ما استلمته بيومين ينفع أخد الفرق في المحفظة؟",
                "لأ يا فندم، مفيش سياسة لمطابقة الأسعار أو رد فرق السعر بأثر رجعي بعد استلام الطلب {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "عندكم Price Match لو السعر قل بعد التسليم؟",
                "لأ، سياسة المتجر ما بتشملش مطابقة السعر بأثر رجعي بعد تسليم الأوردر حتى لو نزل عليه عرض في اليوم التالي {c0}.",
            ),
            TurnVariant(
                "ar_to_en",
                "The item I received yesterday went on sale today. Can you refund the price difference?",
                "Unfortunately, the store does not offer retroactive price matching or price-difference refunds after delivery, even if a promotion starts the next day {c0}.",
            ),
            TurnVariant(
                "ar_to_en",
                "Do you match lower prices posted right after an order is delivered?",
                "No, retroactive price matching is not offered once an order has been delivered {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_contrast_cod_refusal_limit",
        split="train",
        domain="payments",
        behavior="grounded_single",
        supporting_chunks=(
            "الحسابات التي تسجل 3 حالات رفض استلام متتالية لطلبات الدفع عند الاستلام بدون عذر مقبول يتم تعطيل خيار الدفع عند الاستلام لها لمدة 60 يوماً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "فجأة اختيار الكاش عند الاستلام اختفى من حسابي رغم إن الطلب بـ 600 جنيه، ليه؟",
                "لو الحساب اتسجل عليه 3 حالات رفض استلام متتالية لطلبات الدفع عند الاستلام بدون عذر مقبول، بيتقفل خيار الدفع عند الاستلام لمدة 60 يوم {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "بعد كام مرة رفض أوردر كاش بيتقفل الدفع عند الاستلام وبيقعد قد إيه؟",
                "بيتعطل بعد 3 حالات رفض استلام متتالية بدون عذر مقبول، والمنع بيستمر لمدة 60 يوماً {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هو منع الدفع عند الاستلام بيفضل للأبد؟",
                "لأ، تعطيل الدفع عند الاستلام بسبب 3 حالات رفض متتالية بدون عذر بيستمر لمدة 60 يوماً فقط {c0}.",
            ),
            TurnVariant(
                "en",
                "Why would Cash on Delivery be disabled on an account for 60 days?",
                "Accounts with 3 consecutive unexcused refusals of Cash on Delivery parcels have the COD option disabled for 60 days {c0}.",
            ),
            TurnVariant(
                "en",
                "How many consecutive COD refusals trigger a COD restriction?",
                "Three consecutive unexcused COD delivery refusals disable Cash on Delivery for 60 days {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_contrast_no_false_handoff_seal",
        split="train",
        domain="returns",
        behavior="grounded_single",
        supporting_chunks=(
            "العطور ومستحضرات التجميل المفتوحة غير قابلة للاسترجاع نهائياً، ويُستثنى فقط وصول الزجاجة تالفة أو وجود تقرير طبي رسمي بحساسية جلدية.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "اشتريت برفان وفتحته بس ريحته ما عجبتنيش، ينفع أرجّعه؟",
                "للأسف العطور ومستحضرات التجميل المفتوحة غير قابلة للاسترجاع نهائياً، والاستثناء الوحيد لو الزجاجة وصلت مكسورة أو في تقرير طبي رسمي بحساسية جلدية {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "في حالة واحدة ممكن أرجع مستحضر تجميل مفتوح، إيه هي؟",
                "الحالة الاستثنائية هي وصول الزجاجة تالفة أو تقديم تقرير طبي رسمي يثبت وجود حساسية جلدية {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "الكريم عمل لي حساسية في الجلد ومعايا روشتة وتقرير طبي رسمي، ينفع أرجعه بعد الفتح؟",
                "أيوه في الحالة دي مسموح الاسترجاع لأن الاستثناء بيشمل وجود تقرير طبي رسمي بحساسية جلدية أو وصول العبوة تالفة {c0}.",
            ),
            TurnVariant(
                "en",
                "Can an opened perfume bottle be returned if I dislike the scent?",
                "No, opened perfumes and cosmetics are strictly non-returnable unless the bottle arrived damaged or an official medical report confirms a skin allergy {c0}.",
            ),
            TurnVariant(
                "en",
                "What exceptions allow returning an opened cosmetic product?",
                "Opened cosmetics may only be returned if they arrived damaged or with an official medical report documenting a skin allergy {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_xl_weekend_support_hours",
        split="train",
        domain="policies",
        behavior="grounded_single",
        supporting_chunks=(
            "Live human chat agents are available Saturday to Thursday from 9:00 AM to 11:00 PM Cairo time, and on Fridays from 1:00 PM to 10:00 PM.",
        ),
        variants=(
            TurnVariant(
                "en_to_ar",
                "مواعيد موظفين الشات المباشر يوم الجمعة إيه؟",
                "فريق الدعم المباشر متاح يوم الجمعة من الساعة 1 ظهراً لحد 10 مساءً بتوقيت القاهرة {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "هل خدمة العملاء شغالة يوم السبت الساعة 10 الصبح؟",
                "أيوه، من السبت للخميس موظفين الشات المباشر شغالين من 9 الصبح لـ 11 بالليل بتوقيت القاهرة {c0}.",
            ),
            TurnVariant(
                "en",
                "What are your live support chat hours during the week and on Fridays?",
                "Live human chat runs Saturday through Thursday from 9:00 AM to 11:00 PM Cairo time, and Fridays from 1:00 PM to 10:00 PM {c0}.",
            ),
            TurnVariant(
                "en",
                "Is live agent chat open at 11 AM on Friday?",
                "No, Friday live agent chat opens at 1:00 PM and runs until 10:00 PM Cairo time {c0}.",
            ),
            TurnVariant(
                "en_to_ar",
                "ممكن أكلم موظف الساعة 10:30 بالليل يوم الثلاثاء؟",
                "أيوه متاح، الموظفين شغالين من السبت للخميس لحد الساعة 11 مساءً بتوقيت القاهرة {c0}.",
            ),
        ),
    ),

    # -------------------- HANDOFF: MISSING FACT / OFF-KB --------------------
    ScenarioSpec(
        scenario_group="tr_ho_physical_branches",
        split="train",
        domain="policies",
        behavior="handoff",
        handoff_subtype="missing_fact",
        supporting_chunks=(),
        anchor_chunks=(
            "خزائن الاستلام الذاتية (Pickup Lockers) بتحتفظ بالطلبات المدفوعة مسبقاً لمدة 72 ساعة من وقت وصول رسالة الكود.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عندكم فرع في مدينة نصر أروح أشتري منه بنفسي ولا الشراء أونلاين بس؟",
                "HANDOFF | للأسف معنديش معلومات في السياسات المتاحة عن عناوين الفروع الفعلية، وهحوّل حضرتك فوراً لموظف خدمة العملاء يوضحلك الأمر.",
            ),
            TurnVariant(
                "ar_eg",
                "عنوان المعرض بتاعكم في المعادي فين لو سمحت؟",
                "HANDOFF | المعلومة الخاصة بعناوين الفروع مش متوفرة عندي حالياً، وهحوّل حضرتك لموظف خدمة العملاء يساعدك فوراً.",
            ),
            TurnVariant(
                "ar_eg",
                "في فرع ليكم في الإسكندرية أقدر أعاين الأجهزة فيه؟",
                "HANDOFF | معنديش تفاصيل في المستندات المتاحة عن وجود فروع للمعاينة، وهحوّلك لموظف خدمة العملاء يرد عليك.",
            ),
            TurnVariant(
                "en",
                "Do you have a physical showroom in Nasr City where I can inspect products in person?",
                "HANDOFF | Information about physical showroom locations is not available in the provided policies. Let me transfer you to a support specialist right away.",
            ),
            TurnVariant(
                "en",
                "What is the street address of your Maadi store branch?",
                "HANDOFF | I do not have store branch addresses in the available documentation. I am transferring you to a human support agent to assist.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_phone_hotline_number",
        split="train",
        domain="policies",
        behavior="handoff",
        handoff_subtype="missing_fact",
        supporting_chunks=(),
        anchor_chunks=(
            "Live human chat agents are available Saturday to Thursday from 9:00 AM to 11:00 PM Cairo time, and on Fridays from 1:00 PM to 10:00 PM.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "إيه رقم الخط الساخن أو رقم الواتساب بتاع خدمة العملاء؟",
                "HANDOFF | رقم الخط الساخن أو الواتساب مش مذكور في السياسات المتاحة عندي، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن رقم التليفون المختصر عشان أكلمكم مكالمة صوتية؟",
                "HANDOFF | للأسف رقم الهاتف المختصر مش متوفر عندي في المستندات الحالية، وهحوّل حضرتك لخدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "عندكم إيميل للشكاوى الرسمية أبعت عليه؟",
                "HANDOFF | عنوان البريد الإلكتروني الخاص بالشكاوى مش موجود في المعلومات المتاحة، وهحوّلك فوراً لموظف مختص.",
            ),
            TurnVariant(
                "en",
                "What is your customer support hotline number or WhatsApp contact?",
                "HANDOFF | A hotline or WhatsApp contact number is not listed in the provided policies. Let me connect you with a support specialist.",
            ),
            TurnVariant(
                "en",
                "Can you give me the direct phone number for your call center?",
                "HANDOFF | I do not have a call-center phone number in the available policies. I am transferring you to a human support agent.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_upcoming_restock_date",
        split="train",
        domain="listings",
        behavior="handoff",
        handoff_subtype="missing_fact",
        supporting_chunks=(),
        anchor_chunks=(
            "إضافة منتج إلى سلة المشتريات لا تحجز الكمية ولا تثبّت سعر العرض الترويجي حتى إتمام خطوة الدفع وتأكيد الطلب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "كارت الشاشة RTX 4070 خلصان، هينزل المخزن تاني يوم كام الشهر ده؟",
                "HANDOFF | مواعيد توفر المنتجات أو إعادة تخزينها مش متاحة عندي في المستندات، وهحوّلك لموظف خدمة العملاء يراجعلك معاد التوفر.",
            ),
            TurnVariant(
                "ar_eg",
                "هل الأيفون 15 الأسود هيتوفر الأسبوع الجاي ولا لسه؟",
                "HANDOFF | للأسف معنديش بيانات عن مواعيد رجوع المخزون، وهحوّل حضرتك فوراً لموظف مختص يتابع معاك.",
            ),
            TurnVariant(
                "ar_eg",
                "إمتى الموديل ده هيرجع يتباع تاني؟",
                "HANDOFF | معلومة ميعاد توفر المخزون مش موجودة عندي، وهحوّل حضرتك لموظف خدمة العملاء يرد عليك.",
            ),
            TurnVariant(
                "en",
                "When will the out-of-stock 27-inch OLED monitor be restocked?",
                "HANDOFF | Product restock dates are not available in the provided policies. Let me transfer you to a support agent to check inventory updates.",
            ),
            TurnVariant(
                "en",
                "Will the black wireless headphones come back in stock before Friday?",
                "HANDOFF | I do not have live inventory or restock schedule data. I am transferring you to a human agent right away.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_international_shipping_ksa",
        split="train",
        domain="shipping",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "التوصيل داخل القاهرة والجيزة يتم خلال 24 إلى 48 ساعة عمل، وتبلغ رسوم الشحن 60 جنيهاً للطلبات أقل من 1000 جنيه.",
            "التوصيل لمحافظات الصعيد من المنيا حتى أسوان يستغرق 3 إلى 5 أيام عمل مقابل رسوم شحن 95 جنيهاً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "بتشحنوا للسعودية والإمارات ولا داخل مصر بس، والرسوم الدولية كام؟",
                "HANDOFF | السياسات المتاحة عندي بتغطي الشحن جوه مصر فقط ومفيهاش تفاصيل عن الشحن الدولي للسعودية أو الإمارات، وهحوّل حضرتك فوراً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن أطلب أوردر يتبعت لفرع DHL في الرياض؟",
                "HANDOFF | معلومة الشحن الدولي خارج مصر مش متوفرة في السياسات الحالية، وهحوّلك لموظف خدمة العملاء يجاوبك.",
            ),
            TurnVariant(
                "ar_eg",
                "بتوصلوا لشرم الشيخ ومرسى مطروح وبكام؟",
                "HANDOFF | الفقرات المتاحة عندي بتوضح القاهرة والجيزة والصعيد بس ومفيش ذكر لمرسى مطروح أو شرم الشيخ، وهحوّلك لموظف خدمة العملاء يتأكدلك.",
            ),
            TurnVariant(
                "en",
                "Do you ship internationally to Saudi Arabia or UAE, and what are the rates?",
                "HANDOFF | The available policies only cover domestic Egyptian governorates and do not specify international shipping to Saudi Arabia or UAE. Let me transfer you to a support agent.",
            ),
            TurnVariant(
                "en",
                "How much is shipping to Marsa Matrouh or Sharm El-Sheikh?",
                "HANDOFF | Shipping rates for Marsa Matrouh and Sharm El-Sheikh are not covered in the provided chunks. Let me transfer you to a human support agent.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_custom_gift_wrap_cost",
        split="train",
        domain="orders",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "تفعيل خيار 'طلب هدية' يخفي الأسعار من الفاتورة الورقية المطبوعة داخل الطرد.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "تغليف الهدايا الفاخر مع الكارت المطبوع بيكلف كام زيادة؟",
                "HANDOFF | للأسف سعر خدمة تغليف الهدايا أو الكروت المطبوعة مش مذكور في السياسات المتاحة، وهحوّلك لموظف خدمة العملاء يفيدك.",
            ),
            TurnVariant(
                "ar_eg",
                "بتوفروا علب هدايا بالاسم وبكام؟",
                "HANDOFF | المعلومة دي مش متوفرة في المستندات الحالية، وهحوّل حضرتك لموظف خدمة العملاء للرد عليك.",
            ),
            TurnVariant(
                "en",
                "How much does premium gift wrapping with a printed greeting card cost?",
                "HANDOFF | Pricing for premium gift wrapping and printed cards is not specified in the available policies. Let me connect you with a support specialist.",
            ),
            TurnVariant(
                "en",
                "Do you offer custom ribbon gift boxes and what is the surcharge?",
                "HANDOFF | Gift-box packaging options and fees are not covered in the provided chunks. I am transferring you to a support agent.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن أختار لون ورق التغليف للهدية وبكام الخدمة؟",
                "HANDOFF | تفاصيل خدمة تغليف الهدايا وأسعارها مش متاحة عندي، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_arabic_manual_pdf",
        split="train",
        domain="warranty",
        behavior="handoff",
        handoff_subtype="missing_fact",
        supporting_chunks=(),
        anchor_chunks=(
            "تشمل الأجهزة الإلكترونية ضمان وكيل محلي لمدة 12 شهراً ضد عيوب الصناعة فقط، ولا يغطي سوء الاستخدام أو الكسر أو السوائل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "ماكينة القهوة الموديل ده بيجي معاها كتالوج عربي PDF ولا إنجليزي بس؟",
                "HANDOFF | للأسف مفيش عندي تفاصيل عن مواصفات الكتالوج المرفق لموديل معين، وهحوّل حضرتك لموظف خدمة العملاء يراجعلك مواصفات المنتج.",
            ),
            TurnVariant(
                "ar_eg",
                "هل الجهاز ده بيشتغل على كهرباء 220 فولت و50 هرتز في مصر؟",
                "HANDOFF | المواصفات الكهربائية التفصيلية للموديل ده مش متاحة في السياسات الحالية، وهحوّلك لموظف مختص يتأكدلك.",
            ),
            TurnVariant(
                "en",
                "Does this specific espresso machine include an Arabic user manual PDF?",
                "HANDOFF | Product-specific manual details are not available in the provided chunks. Let me transfer you to a support specialist.",
            ),
            TurnVariant(
                "en",
                "What are the exact box dimensions and wattage of this coffee maker?",
                "HANDOFF | Detailed hardware specifications for individual models are not in the available documentation. Transferring you to a human agent now.",
            ),
            TurnVariant(
                "ar_eg",
                "كم واط استهلاك الماكينة وهل فيها فلتر مرفق في العلبة؟",
                "HANDOFF | المواصفات الفنية ومحتويات العلبة للمنتج ده مش موجودة في المستندات المتاحة، وهحوّلك لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_trap_installment_bank_other",
        split="train",
        domain="payments",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "التقسيط بدون فوائد (0%) متاح لمدة 6 أو 12 شهراً لبطاقات ائتمان CIB والبنك الأهلي المصري وQNB على المشتريات من 1500 جنيه فأكثر.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عندي كارت ائتمان بنك مصر، هيدخل في عرض التقسيط بدون فوائد الـ 12 شهر؟",
                "HANDOFF | المعلومات المتاحة بتذكر CIB والبنك الأهلي وQNB فقط وما فيهاش تفاصيل عن بنك مصر، وهحوّل حضرتك فوراً لموظف خدمة العملاء يتأكدلك.",
            ),
            TurnVariant(
                "ar_eg",
                "ينفع أقسّط 0% ببطاقة بنك الإسكندرية أو HSBC؟",
                "HANDOFF | البنك المذكور مش موجود في قائمة البنوك المتاحة عندي، وهحوّلك لموظف خدمة العملاء يراجعلك القوائم المحدّثة.",
            ),
            TurnVariant(
                "ar_eg",
                "هل كروت الخصم المباشر (Debit) ينفع يتقسط بيها؟",
                "HANDOFF | السياسة المتاحة بتتكلم عن بطاقات ائتمان بنوك محددة ومفيش نص عن بطاقات الخصم المباشر، وهحوّلك لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "en",
                "Does Banque Misr or HSBC qualify for the 12-month 0% installment plan?",
                "HANDOFF | The provided chunks only list CIB, NBE, and QNB and do not cover Banque Misr or HSBC. Let me transfer you to a support agent to confirm.",
            ),
            TurnVariant(
                "en",
                "Can I use my CIB debit card instead of a credit card for 0% installments?",
                "HANDOFF | Debit card eligibility is not confirmed in the provided chunks. Let me transfer you to a human support specialist.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_trap_warranty_appliance_motor",
        split="train",
        domain="warranty",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "تشمل الأجهزة الإلكترونية ضمان وكيل محلي لمدة 12 شهراً ضد عيوب الصناعة فقط، ولا يغطي سوء الاستخدام أو الكسر أو السوائل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "موتور الغسالة أو كمبريسور التلاجة عليه ضمان إضافي 10 سنين غير سنة الجهاز؟",
                "HANDOFF | الوثيقة المتاحة عندي بتذكر ضمان الإلكترونيات الـ 12 شهر فقط ومفيش فيها تفاصيل عن ضمان الموتور أو الكمبريسور الـ 10 سنين، وهحوّل حضرتك لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "لو مديت ضمان الموبايل بسنة زيادة مدفوعة، بيكلف كام؟",
                "HANDOFF | خطط تمديد الضمان المدفوعة وأسعارها مش متوفرة في المستندات الحالية، وهحوّلك لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "en",
                "Does the refrigerator compressor come with a separate 10-year motor warranty?",
                "HANDOFF | The available policy only mentions the standard 12-month electronics warranty and does not confirm a separate compressor warranty. Transferring you to a specialist.",
            ),
            TurnVariant(
                "en",
                "How much does an extended 2-year accidental damage protection plan cost?",
                "HANDOFF | Extended paid protection plans are not detailed in the provided chunks. Let me connect you with a support agent.",
            ),
            TurnVariant(
                "ar_eg",
                "هل بتبيعوا تأمين ضد السرقة وكسر الشاشة؟",
                "HANDOFF | معلومة وجود باقات تأمين ضد السرقة أو الكسر مش متوفرة عندي، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_trap_seller_commission_fashion",
        split="train",
        domain="listings",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "تُحوّل مستحقات البائعين بعد 7 أيام من انتهاء مهلة الاسترجاع المقررة للمشتري مع خصم عمولة منصة موحدة قدرها 8%.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "هل فيه خصم على عمولة الـ 8% لو بعت بأكتر من 500 ألف جنيه في الشهر؟",
                "HANDOFF | السياسة المتاحة بتوضح عمولة موحدة 8% ومفيهاش تفاصيل عن شرائح تخفيض لأحجام البيع الكبيرة، وهحوّلك لفريق خدمة البائعين.",
            ),
            TurnVariant(
                "ar_eg",
                "رسوم التخزين في مستودعات Sanad (Fulfillment) للبائعين بتتحسب بكام للمتر؟",
                "HANDOFF | أسعار خدمة التخزين في المستودعات مش متوفرة في السياسات الحالية، وهحوّلك لموظف خدمة العملاء لمساعدتك.",
            ),
            TurnVariant(
                "en",
                "Do high-volume sellers over 500,000 EGP/month get a discounted commission below 8%?",
                "HANDOFF | The available policy states a flat 8% commission and does not list volume-based tiers. Let me transfer you to a seller support specialist.",
            ),
            TurnVariant(
                "en",
                "What are your warehouse storage fees per cubic meter for seller fulfillment?",
                "HANDOFF | Seller warehouse storage fees are not covered in the provided policies. Transferring you to a support agent now.",
            ),
            TurnVariant(
                "ar_eg",
                "كم غرامة تأخير البائع في تسليم الشحنة للمندوب؟",
                "HANDOFF | غرامات تأخير البائعين مش مذكورة في السياسات المتاحة، وهحوّلك لموظف مختص يوضحلك اللائحة.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_trap_courier_specific_slot",
        split="train",
        domain="shipping",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "يتواصل المندوب هاتفياً قبل الوصول بـ 30 دقيقة، وتُتاح محاولتا توصيل مجاناً قبل إرجاع الطلب تلقائياً إلى المستودع.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "ممكن تضمنولي المندوب ييجي بالظبط بين 9 و11 بالليل عشان بكون في الشغل الصبح؟",
                "HANDOFF | السياسة المتاحة بتوضح اتصال المندوب قبلها بـ 30 دقيقة بس ومفيهاش مواعيد تسليم مسائية مضمونة بالظبط، وهحوّلك لموظف خدمة العملاء يرتب معاك.",
            ),
            TurnVariant(
                "ar_eg",
                "ينفع أطلب التوصيل يوم الجمعة الساعة 10 الصبح؟",
                "HANDOFF | تحديد موعد ساعة معينة أو استثناء توصيل ليوم الجمعة مش متاح في السياسات الحالية، وهحوّلك لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "en",
                "Can you guarantee a strict delivery window between 9 PM and 11 PM?",
                "HANDOFF | Specific evening time-slot guarantees are not specified in the provided policy, which only notes a 30-minute prior call. Transferring you to an agent.",
            ),
            TurnVariant(
                "en",
                "Can I schedule my delivery strictly for 8:00 AM sharp?",
                "HANDOFF | Exact time-slot scheduling is not covered in the available documentation. Let me transfer you to a support specialist.",
            ),
            TurnVariant(
                "ar_eg",
                "لو حولت الأوردر لمكان شغلي في وسط البلد هيوصل قبل الساعة 3؟",
                "HANDOFF | السياسات المتاحة مش بتضمن ساعة وصول محددة قبل 3 عصراً، وهحوّلك لموظف خدمة العملاء ينسّق مع شركة الشحن.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_trap_wallet_limit_other",
        split="train",
        domain="payments",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "الطلبات المدفوعة عبر InstaPay أو المحافظ الإلكترونية المصرية يتم تأكيدها خلال 15 دقيقة بدون أي رسوم إضافية.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "أقصى حد أقدر أدفعه في عملية واحدة بـ Vodafone Cash أو InstaPay كام ألف جنيه؟",
                "HANDOFF | حدود المعاملات اليومية أو القصوى لمحافظ الهاتف وInstaPay مش مذكورة في السياسة الحالية، وهحوّلك لموظف خدمة العملاء يتأكدلك.",
            ),
            TurnVariant(
                "ar_eg",
                "هل Apple Pay أو Google Pay شغالين عندكم؟",
                "HANDOFF | دعم Apple Pay أو Google Pay مش مذكور في المستندات المتاحة، وهحوّلك لموظف خدمة العملاء يرد عليك.",
            ),
            TurnVariant(
                "en",
                "What is the maximum single transaction limit for Vodafone Cash or InstaPay?",
                "HANDOFF | Single-transaction caps for Vodafone Cash or InstaPay are not listed in the provided chunks. Transferring you to a human specialist.",
            ),
            TurnVariant(
                "en",
                "Do you accept Apple Pay or PayPal at checkout?",
                "HANDOFF | Whether Apple Pay or PayPal is supported is not stated in the available policies. Let me transfer you to an agent.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن أدفع بالعملة الأجنبية دولار من بطاقة دولية؟",
                "HANDOFF | تفاصيل الدفع بالعملات الأجنبية أو البطاقات الدولية مش متوفرة عندي، وهحوّلك لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_trap_return_software_opened_pc",
        split="train",
        domain="returns",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "منتجات العناية الشخصية المختومة وسماعات الأذن المفتوحة والأكواد الرقمية المفعّلة لا تقبل الاسترجاع إلا إذا ثبت وجود عيب صناعة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "اشتريت تجميعة كمبيوتر مخصصة واتركبت واتشغل عليها ويندوز، ينفع أرجعها؟",
                "HANDOFF | سياسة إرجاع الأجهزة المجمّعة خصيصاً أو البرامج المثبتة مش واضحة في المستندات المتاحة، وهحوّلك لموظف خدمة العملاء يراجع حالتك.",
            ),
            TurnVariant(
                "ar_eg",
                "هل الكتب والروايات الورقية ليها شروط استرجاع مختلفة عن الـ 14 يوم؟",
                "HANDOFF | القواعد الخاصة بالكتب والمطبوعات مش مذكورة في الفقرات المتاحة، وهحوّلك لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "en",
                "Can I return a custom-built desktop PC after Windows was installed at the store?",
                "HANDOFF | Return eligibility for custom-assembled PCs with pre-installed software is not specified in the provided policies. Let me transfer you to a support specialist.",
            ),
            TurnVariant(
                "en",
                "Are prescription pet foods covered under a separate return rule?",
                "HANDOFF | Policies for pet food returns are not covered in the available documentation. Transferring you to a human agent.",
            ),
            TurnVariant(
                "ar_eg",
                "ترابيزة متفصلة بالطلب مقاس خاص، فيها استرجاع؟",
                "HANDOFF | سياسة المنتجات المفصلة حسب الطلب مش متاحة في الوثائق الحالية، وهحوّلك لموظف خدمة العملاء.",
            ),
        ),
    ),

    # -------------------- HANDOFF: LIVE PRIVATE ORDER / ACCOUNT STATE --------------------
    ScenarioSpec(
        scenario_group="tr_ho_live_order_status",
        split="train",
        domain="orders",
        behavior="handoff",
        handoff_subtype="account_state",
        supporting_chunks=(),
        anchor_chunks=(
            "يمكن لعملاء الشراء كزائر (Guest) متابعة شحناتهم عبر إدخال رقم الطلب ورقم الهاتف المستخدم في الدفع بدون إنشاء حساب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "الأوردر بتاعي رقم #98421 وصل فين دلوقتي ولسه ما اتشحنش ليه؟",
                "HANDOFF | معنديش صلاحية أفتح حالة الطلب رقم #98421 من هنا، وهحوّل حضرتك فوراً لموظف خدمة العملاء يراجعلك الشحنة على السيستم.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن تشوفلي حالة الشحنة رقم #31044 اتأخرت ليه؟",
                "HANDOFF | مش متاح عندي الاطلاع على بيانات تتبع الطلبات الشخصية مباشرة، وهحوّلك لموظف خدمة العملاء يتابع #31044 حالاً.",
            ),
            TurnVariant(
                "ar_eg",
                "هل الأوردر #77512 خرج مع المندوب النهارده؟",
                "HANDOFF | مفيش بيانات حية لحالة الطلب #77512 في السياق المتاح، وهحوّل حضرتك لموظف خدمة العملاء يتأكدلك فوراً.",
            ),
            TurnVariant(
                "en",
                "Where is my order #98421 right now and why hasn't it arrived yet?",
                "HANDOFF | I do not have live access to look up individual order status for #98421. Let me transfer you to a support specialist to check the system.",
            ),
            TurnVariant(
                "en",
                "Can you check if order #55190 has been handed to the courier today?",
                "HANDOFF | Live tracking details for #55190 are not available in my context. I am transferring you to a human agent right away.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_live_refund_receipt",
        split="train",
        domain="returns",
        behavior="handoff",
        handoff_subtype="account_state",
        supporting_chunks=(),
        anchor_chunks=(
            "بعد اعتماد المرتجع، يُرد المبلغ إلى البطاقة البنكية أو InstaPay خلال 5 إلى 7 أيام عمل، بينما يُضاف رصيد المحفظة فوراً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "البنك طالب مني رقم مرجع الاسترداد (ARN) للطلب #5510، ممكن تبعتهولي؟",
                "HANDOFF | استخراج رقم مرجع الاسترداد (ARN) للطلب #5510 بيحتاج مراجعة حساباتك على السيستم، وهحوّلك فوراً لموظف خدمة العملاء يبعتهولك.",
            ),
            TurnVariant(
                "ar_eg",
                "فات 9 أيام عمل والمبلغ بتاع طلب #4102 لسه مرجعش الفيزا، شوفولي حصل إيه!",
                "HANDOFF | مراجعة معاملة الاسترداد الفعلية للطلب #4102 بتتطلب دخول على السيستم المالي، وهحوّلك فوراً لموظف خدمة العملاء يتابعها.",
            ),
            TurnVariant(
                "en",
                "My bank needs the ARN refund reference code for order #5510. Can you pull it up?",
                "HANDOFF | Retrieving your bank refund ARN code for #5510 requires access to account transaction logs. Transferring you to a support agent now.",
            ),
            TurnVariant(
                "en",
                "It has been 10 business days and my refund for #4102 hasn't posted. Please check my transaction.",
                "HANDOFF | Investigating a delayed transaction for order #4102 requires live billing access. Let me connect you with a support specialist.",
            ),
            TurnVariant(
                "ar_eg",
                "بعتولي إيصال التحويل البنكي للمرتجع رقم #8820 لو سمحت.",
                "HANDOFF | إرسال إيصالات التحويل الخاصة بحسابك غير متاح من هنا، وهحوّل حضرتك فوراً لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_live_account_ban",
        split="train",
        domain="account",
        behavior="handoff",
        handoff_subtype="account_state",
        supporting_chunks=(),
        anchor_chunks=(
            "الحسابات التي تسجل 3 حالات رفض استلام متتالية لطلبات الدفع عند الاستلام بدون عذر مقبول يتم تعطيل خيار الدفع عند الاستلام لها لمدة 60 يوماً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "حسابي اتعطل فيه الدفع عند الاستلام مع إني استلمت كل طلباتي، راجعولي حسابي دلوقتي!",
                "HANDOFF | مراجعة سجل حسابك الشخصي ومعرفة سبب تعطيل الدفع عند الاستلام بتحتاج صلاحية على السيستم، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "حسابي مقفول أمنياً ومش عارف أدخل رغم إن معايا الرقم، افتحوهولي!",
                "HANDOFF | فك قفل الحسابات ومراجعة بيانات الأمان الشخصية بيتم عبر فريق الدعم مباشرة، وهحوّلك فوراً لموظف مختص.",
            ),
            TurnVariant(
                "en",
                "COD was disabled on my account even though I never refused an order. Please check my account log.",
                "HANDOFF | Reviewing your personal account log and COD eligibility requires live account access. Let me transfer you to a support agent.",
            ),
            TurnVariant(
                "en",
                "My account is locked and I cannot sign in. Can you unlock it now?",
                "HANDOFF | Account verification and unlocking require live identity checks by a human agent. Transferring you now.",
            ),
            TurnVariant(
                "ar_eg",
                "النظام بيقولي رقم موبايلي محظور، شوفولي ليه لو سمحت.",
                "HANDOFF | فحص حالة حظر رقمك الشخصي على السيستم مش متاح عندي، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_live_listing_moderation",
        split="train",
        domain="listings",
        behavior="handoff",
        handoff_subtype="account_state",
        supporting_chunks=(),
        anchor_chunks=(
            "يُحظر عرض الأدوية بوصفة طبية أو المنتجات المقلدة أو مقويات الشبكة غير المرخصة، وتُحذف الإعلانات المخالفة خلال 24 ساعة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "الإعلان بتاعي رقم #L-772 اترفض من المراجعة وأنا متأكد إنه سليم، اترفض ليه؟",
                "HANDOFF | مراجعة سبب رفض الإعلان #L-772 بتحتاج الدخول على ملف الإعلان في السيستم، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن تعيدوا تفعيل الإعلان #L-905 بعد ما غيرت الصور؟",
                "HANDOFF | إعادة مراجعة وتفعيل الإعلانات الخاصة بحسابك بتتم عبر الفريق المختص، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "en",
                "Why was my listing #L-772 rejected by moderation?",
                "HANDOFF | I cannot inspect moderation notes for specific listing #L-772. Let me transfer you to a support specialist right away.",
            ),
            TurnVariant(
                "en",
                "Can you manually approve my updated listing #L-905 right now?",
                "HANDOFF | Manual listing review requires access to your seller dashboard. Transferring you to a human agent.",
            ),
            TurnVariant(
                "ar_eg",
                "شارة البائع الموثق لسه ما ظهرتش في حسابي رغم إني رفعت السجل التجاري من أسبوع.",
                "HANDOFF | متابعة حالة طلب التوثيق الخاص بحسابك بتحتاج مراجعة داخلية، وهحوّلك فوراً لموظف مختص.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_live_wallet_balance",
        split="train",
        domain="promotions",
        behavior="handoff",
        handoff_subtype="account_state",
        supporting_chunks=(),
        anchor_chunks=(
            "نقاط الولاء تُضاف للحساب بعد مرور 14 يوماً من تسليم الطلب وتنتهي صلاحيتها بعد 180 يوماً من تاريخ الإضافة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "كام رصيد النقاط والفلوس اللي في محفظتي دلوقتي وهتنتهي يوم كام؟",
                "HANDOFF | معنديش صلاحية عرض رصيد محفظتك أو تاريخ انتهاء نقاطك الشخصية، وهحوّل حضرتك لموظف خدمة العملاء يراجعلك الرصيد.",
            ),
            TurnVariant(
                "ar_eg",
                "استخدمت كوبون خصم من نقطي وأنا بلغي الطلب، النقط رجعتلي ولا لأ؟",
                "HANDOFF | التحقق من رجوع النقاط في رصيد حسابك بيتطلب فحص الحساب، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "en",
                "What is my current wallet balance and when do my points expire?",
                "HANDOFF | I cannot access your personal wallet balance or point expiration dates. Let me transfer you to a human support agent.",
            ),
            TurnVariant(
                "en",
                "Did my redeemed loyalty points return to my balance after I cancelled yesterday's order?",
                "HANDOFF | Checking whether points were restored to your account requires live system access. Transferring you now.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن تحسبلي معايا كام نقطة في حسابي؟",
                "HANDOFF | رصيد النقاط الخاص بحسابك مش متاح عندي في السياق، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_live_courier_phone",
        split="train",
        domain="shipping",
        behavior="handoff",
        handoff_subtype="account_state",
        supporting_chunks=(),
        anchor_chunks=(
            "يتواصل المندوب هاتفياً قبل الوصول بـ 30 دقيقة، وتُتاح محاولتا توصيل مجاناً قبل إرجاع الطلب تلقائياً إلى المستودع.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "هاتولي رقم تليفون المندوب اللي معاه شحنتي النهارده عشان أنسق معاه!",
                "HANDOFF | بيانات المندوب المكلّف بشحنتك مش متاحة عندي هنا، وهحوّلك فوراً لموظف خدمة العملاء يتواصل مع شركة الشحن أو يوافيك بالتفاصيل.",
            ),
            TurnVariant(
                "ar_eg",
                "المندوب رن عليا والرقم اتقفل ومش لاقيه، ممكن تدوني رقمه البديل؟",
                "HANDOFF | معنديش وصول لأرقام المناديب المرتبطة بطلبك، وهحوّل حضرتك فوراً لموظف خدمة العملاء يساعدك.",
            ),
            TurnVariant(
                "en",
                "Can you give me the phone number of the courier carrying my parcel today?",
                "HANDOFF | I do not have access to individual courier contact numbers for active deliveries. Transferring you to a support specialist now.",
            ),
            TurnVariant(
                "en",
                "The driver called and hung up—please send me his direct number.",
                "HANDOFF | Courier phone details for your shipment are not accessible to me. Let me connect you with a support agent.",
            ),
            TurnVariant(
                "ar_eg",
                "اسم شركة الشحن اللي واخدة طلبي النهارده إيه؟",
                "HANDOFF | معرفة شركة الشحن المسندة لطلبك بتتطلب فتح تفاصيل الطلب على السيستم، وهحوّلك لموظف خدمة العملاء حالاً.",
            ),
        ),
    ),

    # -------------------- HANDOFF: POLICY EXCEPTIONS & ESCALATIONS --------------------
    ScenarioSpec(
        scenario_group="tr_ho_exc_late_return",
        split="train",
        domain="returns",
        behavior="handoff",
        handoff_subtype="policy_exception",
        supporting_chunks=(),
        anchor_chunks=(
            "يمكن إرجاع المنتجات غير المستخدمة خلال 14 يوماً من تاريخ الاستلام بشرط بقاء التغليف الأصلي سليماً وإرفاق الفاتورة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "أنا عارف إن المدة 14 يوم بس كنت مسافر والنهارده اليوم الـ 25، ممكن استثناء أرجّع المنتج وهو مقفول؟",
                "HANDOFF | طلبات الاستثناء بعد انتهاء مهلة الـ 14 يوم بتحتاج مراجعة مشرف خدمة العملاء، وهحوّل حضرتك فوراً للنظر في طلبك.",
            ),
            TurnVariant(
                "ar_eg",
                "عدا 19 يوم على الأوردر وما فتحتوش بسبب ظروف صحية، عاوز استثناء ترجيع.",
                "HANDOFF | الموافقة على استثناءات بعد انتهاء فترة الاسترجاع مش من صلاحيتي، وهحوّلك فوراً لفريق خدمة العملاء لمراجعة حالتك.",
            ),
            TurnVariant(
                "en",
                "I know the limit is 14 days, but I was hospitalized—can you grant an exception on day 24 for an unopened box?",
                "HANDOFF | Exceptions beyond the 14-day return window require supervisor review. Let me transfer you to a support specialist right away.",
            ),
            TurnVariant(
                "en",
                "Can I get a manual return override for an item delivered 3 weeks ago?",
                "HANDOFF | Manual policy overrides require human supervisor approval. I am transferring you to a support agent.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن تمدولي مهلة الاسترجاع 3 أيام زيادة عشان أنا في سفر؟",
                "HANDOFF | طلبات تمديد مهلة الاسترجاع استثنائياً بتتم من خلال فريق خدمة العملاء، وهحوّل حضرتك فوراً.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_exc_bulk_discount",
        split="train",
        domain="promotions",
        behavior="handoff",
        handoff_subtype="policy_exception",
        supporting_chunks=(),
        anchor_chunks=(
            "يُسمح باستخدام كود خصم واحد فقط لكل طلب ولا يمكن دمجه مع الخصومات الفورية لبطاقات البنوك.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "هشتري 40 كرسي مكتب لشركتنا، ينفع تعملولي خصم خاص 25%؟",
                "HANDOFF | عروض الأسعار والخصومات الخاصة بالكميات الكبيرة بتتنسّق مع فريق المبيعات، وهحوّل حضرتك فوراً لموظف مختص.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن تبعتولي عرض سعر مخصص لـ 15 شاشة؟",
                "HANDOFF | إعداد عروض الأسعار الخاصة بالكميات مش متاح في السياسات العامة، وهحوّلك فوراً للفريق المختص.",
            ),
            TurnVariant(
                "en",
                "We want to order 40 office chairs—can you give us a custom 25% bulk discount?",
                "HANDOFF | Custom volume discounts and quotes must be arranged by our sales team. Let me transfer you to a specialist.",
            ),
            TurnVariant(
                "en",
                "Can you negotiate a special price for 25 laptops for an NGO?",
                "HANDOFF | Custom pricing negotiations require human specialist review. Transferring you right away.",
            ),
            TurnVariant(
                "ar_eg",
                "عاوز خصم زيادة غير الكوبون لأني هشتري بـ 80 ألف جنيه.",
                "HANDOFF | طلبات الخصم الإضافي للطلبات الكبيرة بتحتاج مراجعة فريق المبيعات، وهحوّلك فوراً.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_exc_damaged_seal_claim",
        split="train",
        domain="orders",
        behavior="handoff",
        handoff_subtype="policy_exception",
        supporting_chunks=(),
        anchor_chunks=(
            "يُسمح للعميل بفحص التغليف الخارجي ومطابقة اسم الصنف وعدد القطع على الفاتورة قبل التوقيع، لكن تشغيل الأجهزة الإلكترونية المختومة يتطلب سداد القيمة أولاً مع بقاء ضمان 30 يوماً لعيوب الصناعة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "المندوب ساب الكرتونة عند البواب ومشي ولما طلعت لقيت الصندوق مكسور وجهاز جواه متاخد! لازم تعويض فوراً.",
                "HANDOFF | البلاغات الخاصة بتسليم ناقص أو تضرر أثناء غيابك بتحتاج فتح تحقيق فوري مع شركة الشحن، وهحوّل حضرتك حالاً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "استلمت علبة اللابتوب فاضية فيها طوبة بدل الجهاز! أعمل إيه؟",
                "HANDOFF | دي حالة عاجلة بتحتاج تحقيق فوري ومراجعة سجل الشحنة والوزن، وهحوّل حضرتك فوراً لموظف خدمة العملاء لفتح تذكرة تصعيد.",
            ),
            TurnVariant(
                "en",
                "The courier left the box with my doorman and the laptop inside is missing—I need an urgent investigation!",
                "HANDOFF | Suspected missing items or tampered deliveries require an immediate courier investigation. Let me escalate you to a support specialist right away.",
            ),
            TurnVariant(
                "en",
                "The box arrived empty with a broken security seal!",
                "HANDOFF | Tampered or empty parcel claims must be investigated immediately by our escalations team. Transferring you now.",
            ),
            TurnVariant(
                "ar_eg",
                "البواب مضى بدالي والعلبة مفتوحة ومش كاملة!",
                "HANDOFF | حالات الاستلام المثير للشك أو النقص في محتويات الطرد بتتطلب تحقيق فوري من خدمة العملاء، وهحوّلك حالاً.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_exc_chargeback_dispute",
        split="train",
        domain="payments",
        behavior="handoff",
        handoff_subtype="policy_exception",
        supporting_chunks=(),
        anchor_chunks=(
            "في حالة فشل عملية الدفع بالبطاقة وسحب مبلغ معلق، يقوم البنك المُصدر بفك الحجز تلقائياً خلال 3 إلى 5 أيام عمل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "المبلغ اتسحب مرتين بالفعل ودخل في كشف الحساب المؤكد مش معلق وهعمل Chargeback من البنك!",
                "HANDOFF | حالات السحب المكرر المؤكد والنزاعات البنكية بتتطلب مراجعة القسم المالي فوراً، وهحوّلك حالاً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "ar_eg",
                "اتخصم مني المبلغ مرتين برقمين مرجعيين مختلفين ونزلوا كعمليتين نهائيتين، حلولي المشكلة!",
                "HANDOFF | فحص المعاملات المالية المكررة وتنسيق رد المبلغ الزيادة بيتم عبر الفريق المالي، وهحوّلك فوراً لموظف مختص.",
            ),
            TurnVariant(
                "en",
                "I was double-charged as two posted transactions on my credit card statement, not pending holds!",
                "HANDOFF | Confirmed duplicate posted charges require urgent finance-team verification. Let me transfer you to a human support specialist right away.",
            ),
            TurnVariant(
                "en",
                "I'm about to file a bank chargeback for an unauthorized card charge on your site.",
                "HANDOFF | Payment disputes and unauthorized charge reports require immediate escalation to our billing team. Transferring you now.",
            ),
            TurnVariant(
                "ar_eg",
                "في عملية سحب غير مصرح بيها اتعملت على كارت البنك بتاعي عندكم!",
                "HANDOFF | بلاغات العمليات غير المصرح بها بتتحوّل فوراً لفريق الأمان والمالية، وهحوّلك حالاً لموظف مختص.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_exc_tax_exemption_ngo",
        split="train",
        domain="payments",
        behavior="handoff",
        handoff_subtype="policy_exception",
        supporting_chunks=(),
        anchor_chunks=(
            "All listed marketplace prices include 14% Value Added Tax (VAT), and the digital tax invoice is emailed upon dispatch.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "إحنا جمعية خيرية معفاة من ضريبة القيمة المضافة الـ 14% ومعانا خطاب إعفاء، نخصم الضريبة إزاي؟",
                "HANDOFF | إجراءات الإعفاء الضريبي للجهات والجمعيات ومراجعة مستندات الإعفاء بتتم عبر الفريق المالي، وهحوّلك فوراً لموظف مختص.",
            ),
            TurnVariant(
                "ar_eg",
                "محتاج أسترد فاتورة خالية من ضريبة الـ 14% لجهة دبلوماسية.",
                "HANDOFF | طلبات الإعفاء من ضريبة القيمة المضافة للجهات الدبلوماسية أو المعفاة بتحتاج مراجعة الفريق المالي، وهحوّلك فوراً.",
            ),
            TurnVariant(
                "en",
                "We are a VAT-exempt NGO with an official tax exemption letter—how do we waive the 14% VAT?",
                "HANDOFF | Tax-exemption documentation and VAT waivers must be processed by our finance team. Let me transfer you to a specialist right away.",
            ),
            TurnVariant(
                "en",
                "Can you issue a zero-VAT invoice for a diplomatic mission purchase?",
                "HANDOFF | Diplomatic zero-VAT invoicing requires manual finance verification. Transferring you to a human agent now.",
            ),
            TurnVariant(
                "ar_eg",
                "إزاي أبعت جواب الإعفاء الضريبي عشان ما تدفعش الـ 14%؟",
                "HANDOFF | إرسال ومراجعة خطابات الإعفاء الضريبي بيتم مع الفريق المالي عن طريق خدمة العملاء، وهحوّلك حالاً.",
            ),
        ),
    ),

    # -------------------- ADDITIONAL GROUNDED TRAIN SCENARIOS --------------------
    ScenarioSpec(
        scenario_group="tr_pay_instapay_wallet",
        split="train",
        domain="payments",
        behavior="grounded_single",
        supporting_chunks=(
            "الطلبات المدفوعة عبر InstaPay أو المحافظ الإلكترونية المصرية يتم تأكيدها خلال 15 دقيقة بدون أي رسوم إضافية.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "لو دفعت بـ Vodafone Cash أو InstaPay بيتخصم عمولة زيادة وهيتأكد إمتى؟",
                "الدفع عبر InstaPay أو المحافظ الإلكترونية المصرية بدون أي رسوم إضافية، والطلب بيتأكد خلال 15 دقيقة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "دفعت على InstaPay من 5 دقايق، بياخد وقت قد إيه عشان يتأكد؟",
                "تأكيد الطلبات المدفوعة بـ InstaPay أو المحافظ الإلكترونية المصرية بيتم خلال 15 دقيقة من الدفع {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل محفظة Orange Cash أو Etisalat Cash فيها رسوم دفع أونلاين؟",
                "لأ، المحافظ الإلكترونية المصرية وInstaPay بتتأكد خلال 15 دقيقة بدون أي رسوم إضافية {c0}.",
            ),
            TurnVariant(
                "en",
                "Are there extra fees when paying via InstaPay or Egyptian mobile wallets, and how fast is confirmation?",
                "InstaPay and Egyptian mobile wallet payments incur zero extra fees and are confirmed within 15 minutes {c0}.",
            ),
            TurnVariant(
                "en",
                "How long does InstaPay payment confirmation take?",
                "Orders paid with InstaPay or local mobile wallets are confirmed within 15 minutes at no extra cost {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_pay_failed_deduction",
        split="train",
        domain="payments",
        behavior="grounded_single",
        supporting_chunks=(
            "في حالة فشل عملية الدفع بالبطاقة وسحب مبلغ معلق، يقوم البنك المُصدر بفك الحجز تلقائياً خلال 3 إلى 5 أيام عمل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "النت فصل وقت الدفع بالفيزا والأوردر ما اتعملش بس اتسحب مبلغ معلق، فلوسي هترجع إمتى؟",
                "طالما العملية فشلت والمبلغ لسه معلق، البنك بتاعك بيفك الحجز تلقائياً خلال 3 إلى 5 أيام عمل {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "اتخصم مبلغ معلق في محاولة دفع مرفوضة، محتاج أعمل حاجة؟",
                "لأ، في حالات فشل الدفع وسحب مبلغ معلق البنك المُصدر بيفك الحجز لوحده خلال 3 لـ 5 أيام عمل {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "كام يوم عمل عشان فك الحجز المعلق من البطاقة بعد فشل الدفع؟",
                "البنك المُصدر للبطاقة بيفك الحجز المعلق تلقائياً خلال 3 إلى 5 أيام عمل {c0}.",
            ),
            TurnVariant(
                "en",
                "My card checkout failed, the order wasn't created, yet a pending hold appeared. When does it clear?",
                "For failed card attempts that leave a pending hold, your issuing bank automatically releases the hold within 3 to 5 business days {c0}.",
            ),
            TurnVariant(
                "en",
                "How many business days until a failed card payment hold drops off?",
                "The issuing bank automatically releases pending holds from failed checkouts within 3 to 5 business days {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_war_seal_tamper",
        split="train",
        domain="warranty",
        behavior="grounded_single",
        supporting_chunks=(
            "فتح الجهاز أو صيانته في مركز غير معتمد يلغي ضمان الوكيل المحلي فوراً.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "وديت الموبايل لمحل صيانة جنب البيت فتحه عشان ينضفه، كده الضمان لسه شغال؟",
                "للأسف لأ، فتح الجهاز أو تصليحه في مركز غير معتمد بيلغي ضمان الوكيل المحلي فوراً {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو غيرت بطارية اللابتوب بره التوكيل بس بقطعة أصلية، الضمان يفضل ساري؟",
                "لأ يا فندم، أي فتح أو صيانة للجهاز في مركز غير معتمد بيلغي ضمان الوكيل المحلي على طول {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل الصيانة عند فني خارجي بتأثر على الضمان؟",
                "أيوه، فتح الجهاز أو صيانته في مركز غير معتمد بيلغي ضمان الوكيل المحلي فوراً {c0}.",
            ),
            TurnVariant(
                "en",
                "Does opening or repairing my device at an unauthorized shop void the warranty?",
                "Yes, opening or servicing a device at an unauthorized repair shop immediately voids the local agent warranty {c0}.",
            ),
            TurnVariant(
                "en",
                "If a third-party shop opened my laptop just to inspect it, is my warranty still valid?",
                "No, opening the device at any unauthorized center immediately voids the local agent warranty {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ord_address_change",
        split="train",
        domain="orders",
        behavior="grounded_single",
        supporting_chunks=(
            "تعديل عنوان التوصيل متاح خلال ساعتين فقط من تأكيد الطلب طالما حالته ما زالت قيد التجهيز، ولا يمكن تغيير المحافظة بعد خروج الشحنة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "لسه عامل الأوردر من نص ساعة، ينفع أغير عنوان التوصيل؟",
                "أيوه ينفع، تعديل العنوان متاح خلال ساعتين من تأكيد الطلب طالما حالته لسه 'قيد التجهيز'، بس ما ينفعش تغيير المحافظة بعد خروج الشحنة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "الشحنة خرجت خلاص وعاوز أحولها من القاهرة لإسكندرية، ممكن؟",
                "للأسف لأ، تعديل العنوان متاح بس خلال ساعتين وهو قيد التجهيز، ولا يمكن تغيير المحافظة بعد خروج الشحنة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "أقدر أعدل العنوان بعد قد إيه من عمل الأوردر؟",
                "التعديل متاح خلال ساعتين فقط من تأكيد الطلب طالما لسه قيد التجهيز {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I update my delivery address 45 minutes after placing an order?",
                "Yes, address changes are permitted within 2 hours of order confirmation while the status is still 'Processing', though governorates cannot be changed after dispatch {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I switch delivery to another governorate once my parcel has shipped?",
                "No, you cannot switch governorates after the shipment has dispatched, and address edits are only allowed within 2 hours while still Processing {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_lst_price_lock",
        split="train",
        domain="listings",
        behavior="grounded_single",
        supporting_chunks=(
            "إضافة منتج إلى سلة المشتريات لا تحجز الكمية ولا تثبّت سعر العرض الترويجي حتى إتمام خطوة الدفع وتأكيد الطلب.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "حطيت سماعة في السلة إمبارح بسعر العرض والنهارده لقيت السعر زاد وخلصت، ليه ما اتحجزتش ليا؟",
                "وضع المنتج في السلة مش بيحجز الكمية ولا بيثبّت سعر العرض لحد ما تكمل الدفع وتأكد الطلب {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل إضافة المنتج للسلة بتحجزه كام ساعة؟",
                "لأ يا فندم، السلة مش بتحجز الكمية أو السعر نهائياً إلا بعد إتمام الدفع وتأكيد الطلب {c0}.",
            ),
            TurnVariant(
                "en",
                "Does leaving an item in my cart reserve the stock or lock the promo price?",
                "No, adding an item to your cart does not reserve inventory or lock promotional pricing until checkout is completed and confirmed {c0}.",
            ),
            TurnVariant(
                "en",
                "Why did the price change on an item sitting in my cart?",
                "Promotional prices and inventory are not locked while items sit in the cart; they lock only upon checkout confirmation {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "إمتى سعر العرض بيتثبت في الأوردر؟",
                "السعر والكمية بيتثبتوا فقط لحظة إتمام خطوة الدفع وتأكيد الطلب، مش بمجرد الإضافة للسلة {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_acc_otp_login",
        split="train",
        domain="account",
        behavior="grounded_single",
        supporting_chunks=(
            "أكواد التحقق المرسلة عبر SMS أو WhatsApp صالحة لمدة 5 دقائق، وبعد 5 محاولات خاطئة يتم قفل المحاولة مؤقتاً لمدة 30 دقيقة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "كود الـ OTP دخلته غلط 5 مرات والحساب وقف، هيفتح إمتى؟",
                "بعد 5 محاولات خاطئة بيتقفل إدخال الكود مؤقتاً لمدة 30 دقيقة كإجراء أمان، ومع العلم الكود الواحد صالح لـ 5 دقايق بس {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "كود التحقق بتاع الواتساب مدته قد إيه قبل ما ينتهي؟",
                "أكواد التحقق المرسلة عبر SMS أو WhatsApp صالحة لمدة 5 دقائق فقط {c0}.",
            ),
            TurnVariant(
                "en",
                "How long are SMS/WhatsApp OTP codes valid, and what happens after 5 wrong tries?",
                "OTP codes expire after 5 minutes, and 5 consecutive wrong attempts trigger a 30-minute temporary cooldown {c0}.",
            ),
            TurnVariant(
                "en",
                "I entered the wrong verification code 5 times—how long is the lockout?",
                "A 30-minute temporary security cooldown applies after 5 incorrect verification attempts {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "كود الـ SMS ما اشتغلش بعد ربع ساعة من وصوله، طبيعي؟",
                "أيوه، لأن صلاحية كود التحقق 5 دقائق بس فمحتاج تطلب كود جديد {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_promo_points_expiry",
        split="train",
        domain="promotions",
        behavior="grounded_single",
        supporting_chunks=(
            "نقاط الولاء تُضاف للحساب بعد مرور 14 يوماً من تسليم الطلب وتنتهي صلاحيتها بعد 180 يوماً من تاريخ الإضافة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "نقاط الولاء بتنزل بعد قد إيه من استلام الأوردر وبتنتهي إمتى؟",
                "نقاط الولاء بتضاف لحسابك بعد 14 يوم من التسليم وبتنتهي صلاحيتها بعد 180 يوم من تاريخ الإضافة {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "استلمت الطلب النهارده ولسه النقط ما نزلتش في حسابي، ليه؟",
                "النقط بتاخد 14 يوم بعد تسليم الطلب عشان تتضاف للحساب، وبعد إضافتها بتفضل صالحة 180 يوم {c0}.",
            ),
            TurnVariant(
                "en",
                "When are loyalty points credited and how long before they expire?",
                "Loyalty points are credited 14 days after delivery and expire 180 days from the date they are credited {c0}.",
            ),
            TurnVariant(
                "en",
                "How many days are loyalty points valid once added?",
                "Once credited (14 days post-delivery), loyalty points remain valid for 180 days {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "صلاحية النقط اللي في حسابي قد إيه؟",
                "صلاحية نقاط الولاء بتكون 180 يوماً من تاريخ إضافتها للحساب {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ret_multi_item_fee",
        split="train",
        domain="returns",
        behavior="grounded_multi",
        supporting_chunks=(
            "عند إرجاع أكثر من منتج من نفس الطلب في شحنة واحدة بسبب تغيير الرأي، يُخصم رسم سحب واحد فقط بقيمة 50 جنيهاً.",
            "يُشترط تقديم طلب الإرجاع لجميع القطع المراد ردها في نفس الطلب قبل جدولة المندوب للاستفادة من الرسم الموحد.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "هرجّع 3 تيشرتات من نفس الأوردر عشان غيرت رأيي، هدفع 50 جنيه على كل قطعة ولا مرة واحدة؟",
                "لما ترجع أكتر من منتج من نفس الطلب في شحنة واحدة بيتخصم رسم سحب واحد بس 50 جنيه {c0}، بشرط تقدم طلب الاسترجاع لكل القطع مع بعض قبل ما يتجدول المندوب {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "نسيت أضيف بنطلون في طلب الإرجاع والمندوب اتجدول خلاص، لسه هاخد الرسم الموحد الـ 50 جنيه؟",
                "للاستفادة من رسم السحب الموحد (50 جنيه)، لازم تقدم طلب الاسترجاع لجميع القطع في نفس الطلب قبل جدولة المندوب {c0}{c1}.",
            ),
            TurnVariant(
                "en",
                "If I return 3 items from the same order together due to changing my mind, do I pay 50 EGP once or three times?",
                "Returning multiple items from the same order in a single shipment incurs only one 50 EGP pickup fee {c0}, provided all items are submitted together before courier scheduling {c1}.",
            ),
            TurnVariant(
                "en",
                "What condition is required to pay only one 50 EGP fee on a multi-item return?",
                "All items must be included in your return submission before the courier pickup is scheduled {c1}, keeping the fee at a single 50 EGP {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "عندي قطعتين عاوز أرجعهم من أوردر واحد، أعملهم طلب واحد عشان ما أدفعش شحنتين؟",
                "أيوه، قدّم طلب الإرجاع للقطعتين مع بعض قبل جدولة المندوب عشان تدفع رسم سحب واحد 50 جنيه فقط {c0}{c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_pay_installment_refund_rule",
        split="train",
        domain="payments",
        behavior="grounded_multi",
        supporting_chunks=(
            "عند إرجاع طلب تم شراؤه بالتقسيط البنكي، يُرد أصل المبلغ إلى بطاقة الائتمان لإغلاق خطة التقسيط لدى البنك.",
            "الرسوم الإدارية بنسبة 3% الخاصة بخطط 18 و24 شهراً لا تُرد في حالة الإرجاع الناتج عن تغيير الرأي.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "اشتريت جهاز بالتقسيط على 24 شهر ورجعته عشان غيرت رأيي، الأقساط هتقف والـ 3% الإدارية هترجع؟",
                "أصل المبلغ بيرجع على بطاقة الائتمان عشان تتقفل خطة التقسيط مع البنك {c0}، لكن الرسوم الإدارية الـ 3% لخطة 24 شهر مش بتترجع لو الإرجاع بسبب تغيير الرأي {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "لما برجع أوردر متقسط على الفيزا الفلوس بتنزل محفظة ولا بترجع الكارت؟",
                "في حالات الشراء بالتقسيط البنكي، أصل المبلغ بيرجع لبطاقة الائتمان نفسها عشان خطة التقسيط تتقفل عند البنك {c0}.",
            ),
            TurnVariant(
                "en",
                "If I return a 24-month installment purchase because I changed my mind, is the 3% admin fee refunded?",
                "The principal amount is refunded to your credit card to close the bank installment plan {c0}, while the 3% admin fee for 18- and 24-month plans is non-refundable on change-of-mind returns {c1}.",
            ),
            TurnVariant(
                "en",
                "How are bank installment orders refunded when returned?",
                "The principal is refunded directly back to your credit card so the bank can close the installment plan {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو رجعت طلب بالتقسيط الـ 6 شهور، المبلغ بيرجع إزاي؟",
                "أصل المبلغ بيرجع مباشرة لبطاقة الائتمان لإغلاق خطة التقسيط لدى البنك {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_lst_escrow_dispute_freeze",
        split="train",
        domain="listings",
        behavior="grounded_multi",
        supporting_chunks=(
            "عند فتح نزاع من المشتري بسبب عدم مطابقة الوصف، تتجمد مستحقات البائع مؤقتاً لحين صدور قرار فريق التحكيم.",
            "يُمهل البائع 48 ساعة عمل لتقديم صور الشحن أو مستندات الإثبات قبل البت النهائي في النزاع.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "المشتري فتح نزاع إن السلعة غير مطابقة للوصف، فلوسي كـ بائع هتتجمد وقدامي وقت قد إيه أرد؟",
                "لما يتفتح نزاع عدم مطابقة الوصف، مستحقاتك بتتجمد مؤقتاً لحين قرار فريق التحكيم {c0}، وقدامك 48 ساعة عمل تقدم فيها صور الشحن أو إثباتاتك قبل الفصل النهائي {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "أمام البائع كام ساعة يرد على نزاع المشتري؟",
                "البائع بيتاح له مهلة 48 ساعة عمل لتقديم صور الشحن أو المستندات المؤيدة قبل القرار النهائي {c1}.",
            ),
            TurnVariant(
                "en",
                "What happens to seller funds during a buyer 'not as described' dispute and how long do I have to respond?",
                "Seller payouts are temporarily frozen until arbitration decides {c0}, and the seller has 48 business hours to submit dispatch photos or proof before a final ruling {c1}.",
            ),
            TurnVariant(
                "en",
                "How long can a seller take to upload shipping evidence in a dispute?",
                "Sellers have 48 business hours to upload shipping photos or supporting evidence before the final decision {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "هل ينفع أسحب فلوسي كـ بائع وفيه نزاع مفتوح على الطلب؟",
                "لأ، مستحقات البائع بتتجمد مؤقتاً أول ما يتفتح نزاع عدم مطابقة للوصف لحد ما يصدر قرار فريق التحكيم {c0}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_war_resale_transfer",
        split="train",
        domain="warranty",
        behavior="grounded_multi",
        supporting_chunks=(
            "ضمان الأجهزة الإلكترونية مرتبط بالرقم التسلسلي للجهاز والفاتورة الأصلية ولا يُلغى بانتقال الملكية لمشترٍ ثانٍ.",
            "استخدام الضمان من المالك الجديد يشترط تقديم نسخة واضحة من الفاتورة الأصلية أو رقم الطلب الأصلي.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "هشتري لابتوب مستعمل من حد كان جايبه من عندكم من 4 شهور، الضمان لسه شغال باسمي وإيه المطلوب؟",
                "الضمان مربوط بالرقم التسلسلي للجهاز والفاتورة الأصلية ومش بيتلغي لو اتباع لحد تاني {c0}، بس وقت طلب الصيانة هتحتاج تقدم صورة واضحة من الفاتورة الأصلية أو رقم الطلب الأصلي {c1}.",
            ),
            TurnVariant(
                "ar_eg",
                "لو بعت الموبايل بتاعي لصاحبي، هيقدر يستخدم الضمان من غير ما أروح معاه؟",
                "أيوه، الضمان مربوط بالسيريال والفاتورة ومش بيتلغي بتغيير المالك {c0}، المهم يكون معاه نسخة واضحة من الفاتورة الأصلية أو رقم الطلب {c1}.",
            ),
            TurnVariant(
                "en",
                "Does the remaining warranty transfer if I buy a second-hand laptop originally purchased from you?",
                "Yes, the warranty follows the device serial number and original invoice rather than the original owner {c0}, provided the new owner presents a clear copy of the original invoice or order ID {c1}.",
            ),
            TurnVariant(
                "en",
                "What does a second-hand buyer need to claim remaining warranty?",
                "The new owner must provide either a clear copy of the original invoice or the original order number {c1}, as warranty stays tied to the serial number {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "الضمان بينتقل للمشتري التاني لو ممعهوش رقم الطلب ولا الفاتورة؟",
                "رغم إن الضمان ما بيتلغيش بانتقال الملكية {c0}، إلا إنه بيشترط تقديم نسخة من الفاتورة الأصلية أو رقم الطلب الأصلي عشان يستخدمه {c1}.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ship_warehouse_pickup_forbidden",
        split="train",
        domain="shipping",
        behavior="grounded_single",
        supporting_chunks=(
            "لا يُسمح للعملاء بالاستلام المباشر من مستودعات الفرز الرئيسية لأسباب أمنية وتشغيلية؛ ويتم التسليم حصراً عبر المندوب أو خزائن الاستلام.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "ينفع أروح بنفسي للمخزن الرئيسي أستلم الأوردر بدل ما أستنى المندوب؟",
                "لأ يا فندم، الاستلام المباشر من مستودعات الفرز الرئيسية مش مسموح لأسباب أمنية وتشغيلية، والتسليم بيكون فقط عن طريق المندوب أو خزائن الاستلام {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "أنا قريب من المخزن في العاشر، ممكن أعدي آخد الشحنة؟",
                "للأسف غير مسموح بالاستلام المباشر من مستودعات الفرز؛ التوصيل متاح فقط عبر المندوب أو Pickup Lockers {c0}.",
            ),
            TurnVariant(
                "en",
                "Can I drive to the central sorting warehouse to pick up my order faster?",
                "No, walk-in pickups at central sorting warehouses are not permitted for security and operational reasons; delivery is exclusively via courier or pickup lockers {c0}.",
            ),
            TurnVariant(
                "en",
                "Do you allow self-collection from the main fulfillment warehouse?",
                "No, direct customer pickup from sorting warehouses is prohibited; orders must arrive via courier or pickup lockers {c0}.",
            ),
            TurnVariant(
                "ar_eg",
                "إيه طرق التسليم المعتمدة بدل الاستلام من المخزن؟",
                "التسليم بيتم حصراً عبر المندوب أو خزائن الاستلام (Pickup Lockers) وممنوع الاستلام من مستودعات الفرز {c0}.",
            ),
        ),
    ),

    # -------------------- ADDITIONAL HANDOFF TRAIN SCENARIOS --------------------
    ScenarioSpec(
        scenario_group="tr_ho_missing_career_jobs",
        split="train",
        domain="policies",
        behavior="handoff",
        handoff_subtype="missing_fact",
        supporting_chunks=(),
        anchor_chunks=(
            "Live human chat agents are available Saturday to Thursday from 9:00 AM to 11:00 PM Cairo time, and on Fridays from 1:00 PM to 10:00 PM.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عاوز أقدم على شغل مندوب توصيل أو خدمة عملاء معاكم، أبعت الـ CV فين؟",
                "HANDOFF | طلبات التوظيف وإيميلات التقديم مش متاحة عندي في سياسات خدمة العملاء، وهحوّل حضرتك لموظف يساعدك في التواصل مع الموارد البشرية.",
            ),
            TurnVariant(
                "en",
                "Where can I apply for a software engineering internship or customer support job at Sanad?",
                "HANDOFF | Hiring and recruitment channels are not covered in the customer-support documentation. Let me transfer you to an agent to direct your inquiry.",
            ),
            TurnVariant(
                "ar_eg",
                "عندكم وظايف فاضية في المخازن؟",
                "HANDOFF | معلومات الوظائف المتاحة والتقديم غير موجودة في مستندات الدعم عندي، وهحوّلك لموظف مختص.",
            ),
            TurnVariant(
                "en",
                "How do logistics partners register a delivery fleet with your company?",
                "HANDOFF | Fleet partnership onboarding details are not available in my context. Transferring you to a human specialist.",
            ),
            TurnVariant(
                "ar_eg",
                "ممكن إيميل الـ HR عشان أقدم على وظيفة؟",
                "HANDOFF | بيانات التواصل مع قسم الموارد البشرية مش متوفرة عندي، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_trap_pickup_locker_map",
        split="train",
        domain="shipping",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "Prepaid orders routed to Sanad Pickup Lockers are held for 72 hours and require the SMS OTP code or original national ID for collection.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "فين أقرب Pickup Locker لمحطة مترو الدقي أو الزمالك؟",
                "HANDOFF | خريطة أو عناوين خزائن الاستلام المحددة حسب المنطقة مش متوفرة في الفقرات المتاحة، وهحوّل حضرتك لموظف خدمة العملاء يبعتلك أقرب نقطة.",
            ),
            TurnVariant(
                "ar_eg",
                "هل خزينة المعادي فيها أماكن فاضية النهارده لشحنة كبيرة؟",
                "HANDOFF | السعة المتاحة أو مقاسات الخزائن في فرع معين مش متوفرة عندي، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "en",
                "Which Sanad Pickup Locker is closest to Dokki metro station?",
                "HANDOFF | Specific pickup locker locations and addresses are not included in the provided policy chunks. Transferring you to a support agent.",
            ),
            TurnVariant(
                "en",
                "What are the maximum box dimensions a Sanad Pickup Locker compartment can hold?",
                "HANDOFF | Locker compartment dimensions are not stated in the available chunks. Let me connect you with a human support agent.",
            ),
            TurnVariant(
                "ar_eg",
                "أقصى مقاس للكرتونة عشان تدخل في اللوكر كام سنتيمتر؟",
                "HANDOFF | الأبعاد المسموحة لصناديق اللوكر مش مذكورة في المستندات المتاحة، وهحوّلك لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_trap_valu_interest_rate",
        split="train",
        domain="payments",
        behavior="handoff",
        handoff_subtype="adversarial_overlap",
        supporting_chunks=(),
        anchor_chunks=(
            "الدفع عبر ValU يتطلب حساباً مفعّلاً وحد أدنى للسلة 600 جنيه، ويتم تأكيد المعاملة برمز OTP يرسل إلى رقم الهاتف المسجل لدى ValU.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "نسبة الفائدة الشهرية في ValU لو قسطت على 36 شهر كام في المية؟",
                "HANDOFF | نسب الفائدة أو خطط الشهور الخاصة بشركة ValU مش مذكورة في السياسة المتاحة، وهحوّل حضرتك لموظف خدمة العملاء يوضحلك أو يوجهك للخدمة.",
            ),
            TurnVariant(
                "ar_eg",
                "الحد الائتماني بتاعي في ValU اترفض ومش راضي يكمل الدفع، ليه؟",
                "HANDOFF | سبب رفض الحد الائتماني على حسابك في ValU محتاج مراجعة أو تواصل مع الدعم المختص، وهحوّلك فوراً.",
            ),
            TurnVariant(
                "en",
                "What is ValU's exact monthly interest rate for a 36-month financing plan?",
                "HANDOFF | ValU's specific interest rates and tenure schedules are not provided in these chunks. Let me transfer you to a support agent.",
            ),
            TurnVariant(
                "en",
                "Does Souhoola or Sympl work with the same 600 EGP minimum cart?",
                "HANDOFF | Eligibility and minimums for Souhoola or Sympl are not mentioned in the provided documentation. Transferring you to an agent.",
            ),
            TurnVariant(
                "ar_eg",
                "هل بتقبلوا الدفع بـ Sympl أو Contact Now ولا ValU بس؟",
                "HANDOFF | توفر خدمات Sympl أو Contact Now مش مذكور في السياسات المتاحة عندي، وهحوّلك لموظف خدمة العملاء يتأكدلك.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_live_invoice_pdf_edit",
        split="train",
        domain="orders",
        behavior="handoff",
        handoff_subtype="account_state",
        supporting_chunks=(),
        anchor_chunks=(
            "يمكن للعملاء اختيار لغة الفاتورة وإشعارات البريد الإلكتروني (عربي أو إنجليزي) من تبويب التفضيلات في الملف الشخصي.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "عاوز أعدل اسم الشركة والرقم الضريبي على فاتورة الطلب #6612 اللي اتشحن إمبارح، أعمل إيه؟",
                "HANDOFF | تعديل البيانات الضريبية على فاتورة صادرة للطلب #6612 بيتطلب مراجعة الفريق المختص على السيستم، وهحوّلك فوراً.",
            ),
            TurnVariant(
                "ar_eg",
                "الفاتورة بتاعة أوردر #2290 ما وصلتش على الإيميل، ابعتوهالي تاني لو سمحت.",
                "HANDOFF | إعادة إرسال فاتورة الطلب #2290 على بريدك بتحتاج فتح الطلب في السيستم، وهحوّل حضرتك لموظف خدمة العملاء.",
            ),
            TurnVariant(
                "en",
                "Please resend the PDF tax invoice for order #2290 and add our company tax registration number.",
                "HANDOFF | Reissuing or modifying an issued tax invoice for order #2290 requires billing system access. Transferring you to a support specialist.",
            ),
            TurnVariant(
                "en",
                "Why did my invoice for #6612 print with the wrong billing name?",
                "HANDOFF | Inspecting and correcting invoice billing details for #6612 requires human billing support. Transferring you now.",
            ),
            TurnVariant(
                "ar_eg",
                "محتاج نسخة مختومة من الفاتورة الضريبية لأوردر #2290.",
                "HANDOFF | طلب نسخ فواتير مختومة لطلب محدد بيتم عن طريق الفريق المختص، وهحوّلك فوراً لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_exc_warranty_out_of_period",
        split="train",
        domain="warranty",
        behavior="handoff",
        handoff_subtype="policy_exception",
        supporting_chunks=(),
        anchor_chunks=(
            "تشمل الأجهزة الإلكترونية ضمان وكيل محلي لمدة 12 شهراً ضد عيوب الصناعة فقط، ولا يغطي سوء الاستخدام أو الكسر أو السوائل.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "الضمان بتاعي خلص من أسبوعين بس والمشكلة ظهرت امبارح، ممكن تقبلوه صيانة مجانية استثنائية؟",
                "HANDOFF | طلبات الصيانة المجانية الاستثنائية بعد انتهاء فترة الضمان بتحتاج مراجعة مشرف خدمة العملاء، وهحوّل حضرتك فوراً.",
            ),
            TurnVariant(
                "ar_eg",
                "عاوز أعرف تكلفة تغيير شاشة مكسورة على حسابي بره الضمان في التوكيل كام جنيه؟",
                "HANDOFF | تسعير قطع الغيار والصيانة المدفوعة خارج الضمان مش متاح في المستندات عندي، وهحوّلك لموظف خدمة العملاء يوصلك بالتوكيل أو يرد عليك.",
            ),
            TurnVariant(
                "en",
                "My 12-month warranty expired 10 days ago—can I get a goodwill free repair exception?",
                "HANDOFF | Goodwill repair exceptions outside the 12-month warranty require supervisor approval. Let me transfer you to a support specialist.",
            ),
            TurnVariant(
                "en",
                "How much is out-of-warranty screen replacement for a cracked laptop panel?",
                "HANDOFF | Out-of-warranty spare part pricing is not available in the provided policies. Transferring you to a human agent.",
            ),
            TurnVariant(
                "ar_eg",
                "لو هاصلّح الجهاز بره الضمان على حسابي في المركز المعتمد هيكلفني كام؟",
                "HANDOFF | تكاليف الصيانة المدفوعة خارج الضمان مش مذكورة في السياسات الحالية، وهحوّلك لموظف خدمة العملاء.",
            ),
        ),
    ),
    ScenarioSpec(
        scenario_group="tr_ho_trap_fraud_phishing",
        split="train",
        domain="account",
        behavior="handoff",
        handoff_subtype="policy_exception",
        supporting_chunks=(),
        anchor_chunks=(
            "أكواد التحقق المرسلة عبر SMS أو WhatsApp صالحة لمدة 5 دقائق، وبعد 5 محاولات خاطئة يتم قفل المحاولة مؤقتاً لمدة 30 دقيقة.",
        ),
        variants=(
            TurnVariant(
                "ar_eg",
                "جالي رسالة بتقولي كسبت جايزة وطلبوا مني كود الـ OTP وبيانات الكارت، وأنا اديتهم الكود! الحقوني!",
                "HANDOFF | دي حالة احتيال أمني عاجلة بتتطلب تجميد الحساب فوراً وتدخل فريق الأمان، وهحوّل حضرتك حالاً لموظف مختص.",
            ),
            TurnVariant(
                "en",
                "Someone claiming to be Sanad support asked for my OTP and card CVV, and I accidentally shared it—help!",
                "HANDOFF | Suspected phishing and account compromise require immediate emergency action by our security team. Transferring you right now.",
            ),
            TurnVariant(
                "ar_eg",
                "في أوردر اتعمل من حسابي وأنا ما طلبتوش خالص وفيه سحب من كارتتي!",
                "HANDOFF | بلاغات اختراق الحساب والطلبات الاحتيالية بتتحوّل فوراً لفريق الأمان لتجميد الحساب والطلب، وهحوّلك حالاً.",
            ),
            TurnVariant(
                "en",
                "Unauthorized orders just appeared on my account from another city—freeze my account immediately!",
                "HANDOFF | Unauthorized account activity requires immediate intervention by our security team. Escalating you right away.",
            ),
            TurnVariant(
                "ar_eg",
                "حد غير الإيميل والباسورد بتوع حسابي ومش عارف أدخل!",
                "HANDOFF | استرداد الحسابات المخترقة بيتطلب تحقق عاجل من فريق الأمان، وهحوّل حضرتك فوراً لموظف مختص.",
            ),
        ),
    ),
)

